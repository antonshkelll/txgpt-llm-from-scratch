"""
TxGPT: свой GPT для категоризации банковских операций.

Пайплайн:
  1. BPE-токенизатор с нуля на описаниях операций.
  2. Предобучение (pretraining) языковой модели на НЕРАЗМЕЧЕННЫХ описаниях:
     модель учится предсказывать следующий токен, то есть «понимать язык выписок».
  3. Дообучение (fine-tuning) в генеративном формате:  <bos> описание <sep> категория <eos>.
     Классификация = выбрать категорию с максимальной вероятностью P(категория | описание).
  4. Сравнение с бейзлайном (TF-IDF по символьным n-граммам + логистическая регрессия)
     при дефиците разметки (1 000 меток) и при достатке (30 000 меток).

Метрики продукта: accuracy / macro-F1 на знакомых и НЕЗНАКОМЫХ мерчантах,
покрытие автокатегоризации при точности ≥ 95%, задержка на CPU, размер модели.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data import ALL_CATEGORIES, build_dataset
from model import GPT, GPTConfig
from tokenizer import BOS, EOS, PAD, SEP, BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
CKPT, REPORTS = ROOT / "checkpoints", ROOT / "reports"
CKPT.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
torch.set_num_threads(4)
SEED = 0
MAX_TEXT = 40
BLOCK = 64


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------- подготовка последовательностей ----------
def lm_batch(tok, texts):
    """Предобучение: <bos> текст <eos>, лосс на всех токенах."""
    seqs = [[BOS] + tok.encode(t)[:MAX_TEXT] + [EOS] for t in texts]
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), PAD); y = torch.full((len(seqs), L), -100)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
        y[i, :len(s) - 1] = torch.tensor(s[1:])
    return x, y


def cls_seqs(tok, texts, cats):
    """Дообучение/оценка: <bos> текст <sep> категория <eos>; лосс только на категории и <eos>."""
    seqs, starts = [], []
    for t, c in zip(texts, cats):
        prefix = [BOS] + tok.encode(t)[:MAX_TEXT] + [SEP]
        seqs.append(prefix + tok.encode(c) + [EOS]); starts.append(len(prefix))
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), PAD); y = torch.full((len(seqs), L), -100)
    for i, (s, st) in enumerate(zip(seqs, starts)):
        x[i, :len(s)] = torch.tensor(s)
        y[i, st - 1:len(s) - 1] = torch.tensor(s[st:])
    return x, y


def train_loop(model, make_batch, data, steps, lr, bs=64, log_every=250, tag="", val_fn=None):
    """AdamW + warmup + cosine. Если задан val_fn — early stopping: в конце возвращаем лучшие веса по валидации."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    model.train(); t0 = time.time(); hist = []
    best = (-1.0, None, 0)
    for step in range(steps):
        batch = random.sample(data, min(bs, len(data)))
        x, y = make_batch(batch)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(loss.item())
        if (step + 1) % log_every == 0:
            msg = f"  [{tag}] step {step + 1}/{steps}  loss {np.mean(hist[-log_every:]):.3f}"
            if val_fn is not None:
                model.eval(); acc = val_fn(model); model.train()
                msg += f"  val_acc {acc:.3f}"
                if acc > best[0]:
                    best = (acc, {k: v.detach().clone() for k, v in model.state_dict().items()}, step + 1)
            print(msg + f"  ({time.time() - t0:.0f}s)")
    if best[1] is not None:
        model.load_state_dict(best[1])
        print(f"  [{tag}] best val_acc {best[0]:.3f} at step {best[2]}")
    model.eval()
    return hist


@torch.no_grad()
def perplexity(model, tok, texts):
    x, y = lm_batch(tok, texts)
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
    return math.exp(loss.item())


@torch.no_grad()
def category_logprobs(model, tok, texts, chunk=128):
    """log P(категория | текст) для всех категорий → матрица (N, K)."""
    K = len(ALL_CATEGORIES); out = []
    for i in range(0, len(texts), chunk):
        part = texts[i:i + chunk]
        rep_t = [t for t in part for _ in range(K)]
        rep_c = ALL_CATEGORIES * len(part)
        x, y = cls_seqs(tok, rep_t, rep_c)
        logp = F.log_softmax(model(x), -1)
        mask = y != -100
        tok_lp = logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
        out.append(tok_lp.sum(1).view(len(part), K))
    return torch.cat(out).numpy()


def evaluate(pred, proba, gold):
    gold_i = np.array([ALL_CATEGORIES.index(g) for g in gold])
    return dict(accuracy=round(accuracy_score(gold_i, pred), 4),
                macro_f1=round(f1_score(gold_i, pred, average="macro"), 4),
                coverage_at_95=round(coverage_at_precision(proba.max(1), pred == gold_i, 0.95), 4))


def coverage_at_precision(conf, correct, target):
    """Какую долю операций можно категоризировать автоматически, сохранив точность ≥ target."""
    order = np.argsort(-conf)
    prec = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.where(prec >= target)[0]
    return float((ok.max() + 1) / len(conf)) if len(ok) else 0.0


def gpt_predict(model, tok, texts):
    lp = category_logprobs(model, tok, texts)
    proba = np.exp(lp - lp.max(1, keepdims=True)); proba /= proba.sum(1, keepdims=True)
    return proba.argmax(1), proba


def baseline(train, tests):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, min_df=1, sublinear_tf=True)
    X = vec.fit_transform([t for t, _ in train])
    y = [ALL_CATEGORIES.index(c) for _, c in train]
    clf = LogisticRegression(max_iter=3000, C=10).fit(X, y)
    res = {}
    for name, data in tests.items():
        proba = clf.predict_proba(vec.transform([t for t, _ in data]))
        full = np.zeros((len(data), len(ALL_CATEGORIES))); full[:, clf.classes_] = proba
        res[name] = evaluate(full.argmax(1), full, [c for _, c in data])
    return res


def main():
    seed_all()
    t_start = time.time()
    train_all, test_seen, test_unseen, pretrain = build_dataset(30_000, seed=SEED)
    tests = dict(seen=test_seen, unseen=test_unseen)
    print(f"labeled={len(train_all):,} pretrain_texts={len(pretrain):,} test_seen={len(test_seen):,} test_unseen={len(test_unseen):,}")

    tok = BPETokenizer().train(pretrain[:40_000] + ALL_CATEGORIES * 50, vocab_size=1200)
    tok.save(CKPT / "tokenizer.json")
    sample = test_seen[0][0]
    print(f"BPE vocab={len(tok)} | «{sample}» → {[tok.vocab[i] for i in tok.encode(sample)]}")

    cfg = GPTConfig(vocab_size=len(tok), block_size=BLOCK, n_layer=4, n_head=4, n_embd=192, dropout=0.1)
    results = dict(config=cfg.__dict__, budgets={})

    # ---- 1. предобучение ----
    seed_all()
    base = GPT(cfg)
    print(f"GPT params: {base.n_params() / 1e6:.2f}M")
    ppl_before = perplexity(base, tok, [t for t, _ in test_seen[:500]])
    hist = train_loop(base, lambda b: lm_batch(tok, b), pretrain, steps=3000, lr=2e-3, tag="pretrain")
    results["pretrain"] = dict(
        steps=3000, final_loss=round(float(np.mean(hist[-100:])), 3),
        ppl_random_init=round(ppl_before, 1),
        ppl_seen=round(perplexity(base, tok, [t for t, _ in test_seen[:500]]), 2),
        ppl_unseen=round(perplexity(base, tok, [t for t, _ in test_unseen[:500]]), 2))
    torch.save(base.state_dict(), CKPT / "pretrained.pt")
    print("pretrain:", results["pretrain"])

    # генерация: модель «пишет» выписку сама
    seed_all(1)
    gen = base.generate(torch.full((8, 1), BOS), max_new=30, stop_token=EOS)
    results["samples"] = [tok.decode(g.tolist()) for g in gen]
    print("samples:", *results["samples"], sep="\n  ")

    # ---- 2. дообучение при разных объёмах разметки ----
    for budget in (1_000, 30_000):
        # валидация — 15% от бюджета разметки (не больше 1 000), тест не трогаем до конца
        n_val = min(1_000, int(budget * 0.15))
        train, val = train_all[:budget - n_val], train_all[budget - n_val:budget]
        val_texts, val_gold = [t for t, _ in val], np.array([ALL_CATEGORIES.index(c) for _, c in val])
        val_fn = lambda m: float((gpt_predict(m, tok, val_texts)[0] == val_gold).mean())
        res = dict(baseline_tfidf_logreg=baseline(train, tests))
        steps, every = (600, 50) if budget <= 1_000 else (2500, 250)
        for variant in ("gpt_from_scratch", "gpt_pretrained"):
            seed_all()
            model = GPT(cfg)
            if variant == "gpt_pretrained":
                model.load_state_dict(torch.load(CKPT / "pretrained.pt"))
            train_loop(model, lambda b: cls_seqs(tok, [t for t, _ in b], [c for _, c in b]),
                       train, steps=steps, lr=5e-4 if variant == "gpt_pretrained" else 1e-3,
                       log_every=every, tag=f"{variant}/{budget}", val_fn=val_fn)
            res[variant] = {}
            for name, data in tests.items():
                pred, proba = gpt_predict(model, tok, [t for t, _ in data])
                res[variant][name] = evaluate(pred, proba, [c for _, c in data])
            if variant == "gpt_pretrained" and budget == 30_000:
                torch.save(model.state_dict(), CKPT / "txgpt_classifier.pt")
                t0 = time.perf_counter()
                gpt_predict(model, tok, [t for t, _ in test_seen[:256]])
                results["latency_ms_per_tx_cpu"] = round((time.perf_counter() - t0) / 256 * 1000, 2)
        results["budgets"][budget] = res
        print(f"budget={budget}:", json.dumps(res, ensure_ascii=False))

    results["n_params"] = base.n_params()
    results["train_time_min"] = round((time.time() - t_start) / 60, 1)
    (REPORTS / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
