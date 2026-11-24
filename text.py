"""Text preprocessing for Part 3: our own tokenizer and a word-level vocabulary.

    from text import Vocabulary, tokenize
    vocab = Vocabulary.from_captions(data["train"]["captions"], min_count=5)
    ids = vocab.encode("A dog on a couch.")    # [<bos>, a, dog, on, a, couch, ., <eos>]
    vocab.decode(ids)                          # ["a", "dog", "on", "a", "couch", "."]

`vocab.encode` is the `encode` function that `data.caption_loaders` expects, and `vocab.pad_id`
its `pad_id`.
"""

import re
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)   # PAD gets id 0

# A token is either a word (letters/digits, optionally with one apostrophe part such as
# "man's" or "don't") or a single punctuation character. Everything is lowercased first, so
# "Semi-truck." becomes ["semi", "-", "truck", "."].
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\sa-z0-9]")


def tokenize(caption):
    """Lowercase, separate punctuation, split on whitespace."""
    return _TOKEN.findall(caption.lower())


class Vocabulary:
    """Word-level vocabulary built from the training captions only.

    Words that occur fewer than `min_count` times in the training captions are mapped to <unk>.
    Ids: 0 = <pad>, 1 = <bos>, 2 = <eos>, 3 = <unk>, then the words by decreasing frequency
    (ties alphabetically, so the vocabulary is deterministic).
    """

    def __init__(self, words, counts=None, min_count=None):
        self.itos = list(SPECIALS) + list(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = counts            # Counter of all training tokens (for statistics)
        self.min_count = min_count
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = (self.stoi[s] for s in SPECIALS)

    @classmethod
    def from_captions(cls, captions, min_count=5):
        """captions: list (images) of lists (5 per image) of caption strings."""
        counts = Counter(tok for image_captions in captions for c in image_captions
                         for tok in tokenize(c))
        words = sorted((w for w, n in counts.items() if n >= min_count),
                       key=lambda w: (-counts[w], w))
        return cls(words, counts, min_count)

    def __len__(self):
        return len(self.itos)

    def encode(self, caption):
        """Caption string -> [<bos>, token ids ..., <eos>] (unknown words -> <unk>)."""
        return ([self.bos_id] + [self.stoi.get(tok, self.unk_id) for tok in tokenize(caption)]
                + [self.eos_id])

    def decode(self, ids, strip_special=True):
        """Token ids -> list of words. Stops at the first <eos>; with strip_special, <bos> and
        <pad> are dropped (but <unk> is kept, since it is a generated word)."""
        words = []
        for i in (ids.tolist() if hasattr(ids, "tolist") else ids):
            if i == self.eos_id:
                break
            if strip_special and i in (self.bos_id, self.pad_id):
                continue
            words.append(self.itos[i])
        return words
