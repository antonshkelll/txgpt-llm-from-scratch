"""Сводный график: точность моделей на знакомых и незнакомых мерчантах при 1k и 30k меток."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REP = Path(__file__).resolve().parents[1] / "reports"
r = json.loads((REP / "results.json").read_text())["budgets"]
p = json.loads((REP / "probe_results.json").read_text())
models = [("baseline_tfidf_logreg", "TF-IDF + LogReg", "#9aa0a6", r),
          ("gpt_from_scratch", "GPT без предобучения", "#F2CF5B", r),
          ("gpt_pretrained", "TxGPT: предобучение + дообучение", "#E45756", r),
          ("probe_pretrained", "Linear probe на TxGPT", "#4C78A8", p)]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, split, title in [(axes[0], "seen", "Знакомые мерчанты"), (axes[1], "unseen", "Незнакомые мерчанты")]:
    x = np.arange(2)
    for i, (key, label, color, src) in enumerate(models):
        vals = [src[b][key][split]["accuracy"] for b in ("1000", "30000")]
        bars = ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, color=color, label=label)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, ["1 000 меток", "30 000 меток"]); ax.set_title(title); ax.set_ylim(0, 1.08)
axes[0].set_ylabel("accuracy")
axes[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(REP / "accuracy.png")
