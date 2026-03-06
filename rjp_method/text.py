import re
from collections import Counter

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s.,-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str):
    return normalize_text(text).split()


class Vocab:
    def __init__(self, min_freq: int = 3):
        self.min_freq = min_freq
        self.stoi = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
        self.itos = [PAD, BOS, EOS, UNK]

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        for w, c in counter.items():
            if c >= self.min_freq and w not in self.stoi:
                self.stoi[w] = len(self.itos)
                self.itos.append(w)

    def encode(self, text: str, max_len: int):
        ids = [self.stoi[BOS]]
        for tok in tokenize(text):
            ids.append(self.stoi.get(tok, self.stoi[UNK]))
            if len(ids) >= max_len - 1:
                break
        ids.append(self.stoi[EOS])
        pad_id = self.stoi[PAD]
        if len(ids) < max_len:
            ids.extend([pad_id] * (max_len - len(ids)))
        return ids

    def decode(self, ids):
        words = []
        for idx in ids:
            token = self.itos[idx] if idx < len(self.itos) else UNK
            if token in (PAD, BOS):
                continue
            if token == EOS:
                break
            words.append(token)
        return " ".join(words)

    def __len__(self):
        return len(self.itos)
