
from vcpi_ml.data import build_smile_char_vocab, tokenize_smile_char

def test_pad_id_reserved():
    vocab = build_smile_char_vocab(["CCO"])
    assert 0 not in vocab.values()

def test_encode_and_mask_align():
    vocab = build_smile_char_vocab(["CCO", "CN"])
    token, mask = tokenize_smile_char(["CCO"], vocab, max_len=8)
    assert token.shape == (1, 8)
    assert mask[0].sum() == 3
    assert (token[0, 3:] == 0).all()
    assert [vocab[c] for c in "CCO"] == list(token[0, : 3])