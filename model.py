"""Small count language model and a linear grammaticality head; no downloads."""
import math
import re
from collections import Counter

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin


def tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)


def sequences(text):
    return [tokens(text), list(re.sub(r"\s+", " ", text.lower()).strip())]


def key(items):
    return "\x1f".join(items)


def fit_statistics(texts):
    tables = [[Counter() for _ in range(n)] for n in (3, 5)]
    for text in texts:
        for seq, counts in zip(sequences(text), tables):
            seq = ["<s>"] * (len(counts) - 1) + seq + ["</s>"]
            for i in range(len(counts) - 1, len(seq)):
                for n, table in enumerate(counts, 1):
                    table[key(seq[i-n+1:i+1])] += 1
    return [[dict(c) for c in counts] for counts in tables]


def features(text, statistics):
    result = []
    for seq, tables in zip(sequences(text), statistics):
        padded = ["<s>"] * (len(tables) - 1) + seq + ["</s>"]
        total = sum(tables[0].values())
        vocabulary = len(tables[0]) + 1
        for n, table in enumerate(tables, 1):
            surprise = []
            for i in range(len(tables)-1, len(padded)):
                unit = padded[i-n+1:i+1]
                if n == 1:
                    probability = (table.get(key(unit), 0) + 0.1) / (total + 0.1*vocabulary)
                else:
                    # Interpolate with unigram probability to avoid infinite OOV penalties.
                    base = (tables[0].get(unit[-1], 0) + 0.1) / (total + 0.1*vocabulary)
                    context = (tables[0].get("</s>",0) if all(t == "<s>" for t in unit[:-1])
                               else tables[n-2].get(key(unit[:-1]), 0))
                    probability = (table.get(key(unit), 0) + 5*base) / (context + 5)
                surprise.append(-math.log(max(probability, 1e-15)))
            result.extend([float(np.mean(surprise)), float(np.quantile(surprise, 0.9))])
        result.append(sum(t not in tables[0] for t in seq) / max(1, len(seq)))
    return result


class GrammaticalityModel(torch.nn.Module, PyTorchModelHubMixin):
    """Hub format: config.json (count tables) + model.safetensors (linear head).

    Import this class, then call from_pretrained(local_directory_or_hub_id).
    No Transformers AutoModel or hosted inference provider is required/supported.
    """
    def __init__(self, statistics: list, feature_indices: list):
        super().__init__()
        self.statistics = statistics
        self.feature_indices = feature_indices
        size = len(feature_indices)
        self.register_buffer("center", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(size, dtype=torch.float64))
        self.linear = torch.nn.Linear(size, 1, dtype=torch.float64)

    @torch.no_grad()
    def forward(self, texts):
        if not texts:
            return torch.empty(0, dtype=torch.float64)
        x = np.array([features(t, self.statistics) for t in texts])[:, self.feature_indices]
        x = torch.as_tensor(x, dtype=torch.float64)
        return self.linear((x-self.center)/self.scale).squeeze(-1).clamp(0, 1)
