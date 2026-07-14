"""Batch hand-analysis and counterfactual mapping."""

from __future__ import annotations


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
            mapping[key] = len(rows); rows.append(row); melds.append(opened)
        inverse.append(mapping[key])
    return rows, melds, inverse
