import numpy as np


def hand_counts():
    counts = np.zeros(34, np.uint8)
    for tile in (0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28): counts[tile] += 1
    return counts


def event_rows():
    return [{"environment_id": 0, "episode_generation": 1, "sequence": 0,
             "kind": 1, "actor_seat": 255, "visibility_mask": 15}]

