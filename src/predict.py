"""
Демо: категоризация операций и генерация «выписки» обученной моделью.

    python src/predict.py "ВКУСНО И ТОЧКА 0453 МОСКВА" "Оплата СБП ИП Петров (барбершоп)"
    python src/predict.py --generate 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import ALL_CATEGORIES  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import BOS, EOS, BPETokenizer  # noqa: E402
from train import BLOCK, CKPT, gpt_predict  # noqa: E402


def load(name: str):
    tok = BPETokenizer.load(CKPT / "tokenizer.json")
    model = GPT(GPTConfig(vocab_size=len(tok), block_size=BLOCK, dropout=0.0))
    model.load_state_dict(torch.load(CKPT / name))
    return tok, model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("texts", nargs="*")
    p.add_argument("--generate", type=int, default=0)
    a = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):          # консоль Windows по умолчанию не в UTF-8
        sys.stdout.reconfigure(encoding="utf-8")
    if a.generate:
        tok, model = load("pretrained.pt")
        out = model.generate(torch.full((a.generate, 1), BOS), max_new=30, stop_token=EOS)
        for row in out:
            print(" ", tok.decode(row.tolist()))
    if a.texts:
        tok, model = load("txgpt_classifier.pt")
        pred, proba = gpt_predict(model, tok, a.texts)
        for t, k, pr in zip(a.texts, pred, proba):
            top = np.argsort(-pr)[:3]
            alt = ", ".join(f"{ALL_CATEGORIES[i]} {pr[i]:.0%}" for i in top[1:])
            print(f"{t!r:48} → {ALL_CATEGORIES[k]} ({pr[k]:.0%})   | ещё: {alt}")


if __name__ == "__main__":
    main()
