"""
Linear probe: замораживаем предобученный TxGPT и используем его как «извлекатель признаков».
Эмбеддинг операции = среднее скрытых состояний последнего слоя по токенам описания.
Поверх — логистическая регрессия. Плюс гибрид: TF-IDF + эмбеддинги GPT.

Вопрос: научилось ли предобучение на НЕРАЗМЕЧЕННЫХ описаниях чему-то полезному,
когда разметки мало (1 000 меток)?
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data import ALL_CATEGORIES, build_dataset  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import BOS, PAD, BPETokenizer  # noqa: E402
from train import BLOCK, CKPT, MAX_TEXT, REPORTS, SEED, evaluate  # noqa: E402

torch.set_num_threads(4)


@torch.no_grad()
def embed(model, tok, texts, chunk=512):
    out = []
    for i in range(0, len(texts), chunk):
        seqs = [[BOS] + tok.encode(t)[:MAX_TEXT] for t in texts[i:i + chunk]]
        L = max(map(len, seqs))
        x = torch.full((len(seqs), L), PAD)
        mask = torch.zeros(len(seqs), L)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s); mask[j, 1:len(s)] = 1
        h = model.hidden(x)
        out.append(((h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)).numpy())
    return np.vstack(out)


def fit_eval(Xtr, ytr, tests):
    clf = LogisticRegression(max_iter=4000, C=10).fit(Xtr, ytr)
    res = {}
    for name, (X, gold) in tests.items():
        proba = np.zeros((X.shape[0], len(ALL_CATEGORIES)))
        proba[:, clf.classes_] = clf.predict_proba(X)
        res[name] = evaluate(proba.argmax(1), proba, gold)
    return res


def main():
    train_all, test_seen, test_unseen, _ = build_dataset(30_000, seed=SEED)
    tok = BPETokenizer.load(CKPT / "tokenizer.json")
    results = {}
    for budget in (1_000, 30_000):
        n_val = min(1_000, int(budget * 0.15))
        train = train_all[:budget - n_val]                    # тот же train, что и у остальных моделей
        ytr = [ALL_CATEGORIES.index(c) for _, c in train]
        texts = {"train": [t for t, _ in train], "seen": [t for t, _ in test_seen], "unseen": [t for t, _ in test_unseen]}
        gold = {"seen": [c for _, c in test_seen], "unseen": [c for _, c in test_unseen]}

        res = {}
        for name, ckpt in (("probe_random_init", None), ("probe_pretrained", "pretrained.pt")):
            torch.manual_seed(SEED)
            m = GPT(GPTConfig(vocab_size=len(tok), block_size=BLOCK, dropout=0.0)).eval()
            if ckpt:
                m.load_state_dict(torch.load(CKPT / ckpt))
            E = {k: embed(m, tok, v) for k, v in texts.items()}
            sc = StandardScaler().fit(E["train"])
            E = {k: sc.transform(v) for k, v in E.items()}
            res[name] = fit_eval(E["train"], ytr, {k: (E[k], gold[k]) for k in ("seen", "unseen")})
            if ckpt:
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit(texts["train"])
                H = {k: hstack([vec.transform(v), csr_matrix(E[k] * 0.1)]).tocsr() for k, v in texts.items()}
                res["hybrid_tfidf_plus_gpt"] = fit_eval(H["train"], ytr, {k: (H[k], gold[k]) for k in ("seen", "unseen")})
        results[budget] = res
        print(budget, json.dumps(res, ensure_ascii=False))
    (REPORTS / "probe_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
