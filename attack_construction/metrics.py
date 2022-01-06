import numpy as np
import torch


def general_objectness(labels, device):  # as in InvisibleCloak
    scores = []
    for i in range(len(labels["labels"])):
        # and labels["scores"][i] > 0.6
        if labels["labels"][i] == 1:
            scores.append(max(labels["scores"][i] + 1, 0)**2)
    return sum(scores)


# TV - total variation penalty (smooth for patch)
def total_variation(patch):
    # wiki: https://en.wikipedia.org/wiki/Total_variation_denoising
    a1 = patch[:, :, :-1] - patch[:, :, 1:]
    a2 = patch[:, :-1, :] - patch[:, 1:, :]
    return a1.abs().sum() + a2.abs().sum()
