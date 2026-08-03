"""Batch hand-analysis and counterfactual mapping."""

from __future__ import annotations

import numpy as np


class AnalysisBatch:
    """Deduplicate hand queries shared by all consumers of one native frame."""

    def __init__(self):
        self.rows = []
        self.melds = []
        self.mapping = {}

    def add(self, counts, open_melds: int) -> int:
        if isinstance(counts, np.ndarray) and counts.dtype == np.uint8:
            row = counts
        elif isinstance(counts, (bytes, bytearray, memoryview)):
            row = np.frombuffer(counts, dtype=np.uint8)
        else:
            row = np.asarray(counts, dtype=np.uint8)
        if row.shape != (34,):
            raise ValueError("hand-analysis counts must have shape [34]")
        key = (row.tobytes(), int(open_melds))
        index = self.mapping.get(key)
        if index is None:
            index = len(self.rows)
            self.mapping[key] = index
            self.rows.append(row)
            self.melds.append(int(open_melds))
        return index

    def run(self):
        return analyze(self.rows, self.melds) if self.rows else None


def analyze(counts, open_melds):
    import numpy as np
    import riichi

    counts = np.ascontiguousarray(counts, dtype=np.uint8)
    open_melds = np.ascontiguousarray(open_melds, dtype=np.uint8)
    return riichi.analyze_hands(counts, open_melds)


def unique_rows(counts, open_melds):
    mapping, rows, melds = {}, [], []
    inverse = []
    for row, opened in zip(counts, open_melds):
        key = (bytes(row), int(opened))
        if key not in mapping:
            mapping[key] = len(rows)
            rows.append(row)
            melds.append(opened)
        inverse.append(mapping[key])
    return rows, melds, inverse
