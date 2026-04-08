import re


def tokenize_words(text):
    """Split text into lowercase word tokens."""
    return re.findall(r"\w+", str(text).lower())


def build_vocab(lines):
    """Build a token-to-id vocabulary from an iterable of text lines."""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    next_id = 2

    for line in lines:
        for word in tokenize_words(line):
            if word not in vocab:
                vocab[word] = next_id
                next_id += 1

    return vocab


def encode(text, vocab):
    """Convert text into token ids using a vocabulary."""
    unk_id = vocab.get("<UNK>", 1)
    return [vocab.get(token, unk_id) for token in tokenize_words(text)]


def decode(token_ids, id_to_token):
    """Convert token ids back into a space-separated token string."""
    return " ".join(id_to_token.get(token_id, "<UNK>") for token_id in token_ids)
