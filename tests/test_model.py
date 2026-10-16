import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import ALL_CATEGORIES, build_dataset  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402

TEXTS = [t for t, _ in build_dataset(400, seed=1)[0]]


def test_bpe_roundtrip_and_compression():
    tok = BPETokenizer().train(TEXTS, vocab_size=300)
    for t in TEXTS[:50]:
        assert tok.decode(tok.encode(t)) == " ".join(t.split())
    chars = sum(len(t) for t in TEXTS)
    tokens = sum(len(tok.encode(t)) for t in TEXTS)
    assert tokens < 0.75 * chars            # склейки реально сжимают текст (≈0.66 при словаре 300)


def test_bpe_save_load(tmp_path):
    tok = BPETokenizer().train(TEXTS, vocab_size=250)
    tok.save(tmp_path / "t.json")
    tok2 = BPETokenizer.load(tmp_path / "t.json")
    assert tok2.encode(TEXTS[0]) == tok.encode(TEXTS[0])


def test_attention_is_causal():
    """Изменение будущего токена не должно менять предсказания для прошлых позиций."""
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()
    x = torch.randint(0, 50, (1, 10))
    y = x.clone(); y[0, 7] = (y[0, 7] + 1) % 50
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a[0, :7], b[0, :7], atol=1e-6)
    assert not torch.allclose(a[0, 7:], b[0, 7:])


def test_weight_tying():
    m = GPT(GPTConfig(vocab_size=40, block_size=8, n_layer=1, n_head=1, n_embd=16))
    assert m.head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


def test_can_overfit_tiny_batch():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=30, block_size=12, n_layer=2, n_head=2, n_embd=32, dropout=0.0))
    x = torch.randint(0, 30, (4, 12))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        loss = F.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1


def test_dataset_has_all_categories():
    train, seen, unseen, pretrain = build_dataset(3000, seed=0)
    assert {c for _, c in train} == set(ALL_CATEGORIES)
    assert len(pretrain) == 4 * 3000
