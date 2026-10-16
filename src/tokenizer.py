"""
BPE-токенизатор с нуля (как в GPT-2 / SentencePiece, но проще).

Идея: начинаем с символов и много раз склеиваем самую частую пару соседних токенов.
Так частые куски («ПЕРЕ», «ВОД», «МОСКВА») становятся одним токеном, а редкие слова
разбираются на части — неизвестных слов не бывает.

Пробел кодируем маркером «▁» в начале слова, чтобы декодирование было обратимым.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

SPECIAL = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]
PAD, BOS, EOS, SEP, UNK = range(5)


class BPETokenizer:
    def __init__(self, merges: list[tuple[str, str]] | None = None, vocab: list[str] | None = None):
        self.merges = merges or []
        self.vocab = vocab or list(SPECIAL)
        self._rebuild()

    def _rebuild(self):
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._cache: dict[str, list[int]] = {}

    @staticmethod
    def _words(text: str) -> list[str]:
        return ["▁" + w for w in text.split(" ") if w]

    # ---------- обучение ----------
    def train(self, texts: list[str], vocab_size: int) -> "BPETokenizer":
        word_freq = Counter(w for t in texts for w in self._words(t))
        words = [list(w) for w in word_freq]
        freqs = list(word_freq.values())
        chars = sorted({c for w in words for c in w})
        self.vocab = list(SPECIAL) + chars

        pair_count: Counter = Counter()
        where: dict[tuple, set] = defaultdict(set)      # пара → в каких словах встречается
        for i, w in enumerate(words):
            for a, b in zip(w, w[1:]):
                pair_count[(a, b)] += freqs[i]
                where[(a, b)].add(i)

        while len(self.vocab) < vocab_size and pair_count:
            (a, b), cnt = pair_count.most_common(1)[0]
            if cnt < 2:
                break
            new = a + b
            self.merges.append((a, b))
            self.vocab.append(new)
            for i in list(where[(a, b)]):
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):                  # убираем старые пары слова из счётчика
                    pair_count[p] -= f
                    if pair_count[p] <= 0:
                        del pair_count[p]
                j, merged = 0, []
                while j < len(w):
                    if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                        merged.append(new); j += 2
                    else:
                        merged.append(w[j]); j += 1
                words[i] = merged
                for p in zip(merged, merged[1:]):        # и добавляем новые
                    pair_count[p] += f
                    where[p].add(i)
            where.pop((a, b), None)
        self._rebuild()
        return self

    # ---------- кодирование ----------
    def _encode_word(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get(p, 1 << 30), i) for i, p in enumerate(zip(parts, parts[1:]))]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        ids = [self.stoi.get(p, UNK) for p in parts]
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for w in self._words(text) for i in self._encode_word(w)]

    def decode(self, ids: list[int]) -> str:
        toks = [self.vocab[i] for i in ids if i >= len(SPECIAL)]
        return "".join(toks).replace("▁", " ").strip()

    def __len__(self):
        return len(self.vocab)

    # ---------- сохранение ----------
    def save(self, path: Path):
        path.write_text(json.dumps(dict(vocab=self.vocab, merges=self.merges), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls([tuple(m) for m in d["merges"]], d["vocab"])
