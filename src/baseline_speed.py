"""Скорость и размер бейзлайна TF-IDF + LogReg — так же, как замерялся TxGPT в train.py:
пачка из 256 операций, время на одну операцию на CPU. Результат: reports/baseline_speed.json
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data import build_dataset
from train import SEED

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main():
    train_all, test_seen, _, _ = build_dataset(30_000, seed=SEED)
    train = train_all[:29_000]                      # тот же бюджет, что в train.py (30 000 минус валидация)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, min_df=1, sublinear_tf=True)
    X = vec.fit_transform([t for t, _ in train])
    clf = LogisticRegression(max_iter=3000, C=10).fit(X, [c for _, c in train])
    batch = [t for t, _ in test_seen[:256]]
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        clf.predict_proba(vec.transform(batch))
        times.append((time.perf_counter() - t0) / len(batch) * 1000)
    gpt = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    out = dict(tfidf_latency_ms_per_tx=round(float(np.median(times)), 3),
               tfidf_weights=int(clf.coef_.size + clf.intercept_.size),
               txgpt_latency_ms_per_tx=gpt["latency_ms_per_tx_cpu"], txgpt_params=gpt["n_params"])
    out["speedup"] = round(out["txgpt_latency_ms_per_tx"] / out["tfidf_latency_ms_per_tx"])
    (REPORTS / "baseline_speed.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
