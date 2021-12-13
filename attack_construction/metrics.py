import torch


def general_objectness(labels, device):  # as in InvisibleCloak
    score = torch.tensor(0.0).to(device)
    # score.requires_grad = True
    for i in range(len(labels["labels"])):
        if labels["labels"][i] == 1: # and labels["scores"][i] > 0.6
            score += max(labels["scores"][i] + 1, 0)**2
    return score


def total_variation(patch):  # TV - total variation penalthy (smooth for patch)
    a1 = patch[:, :, :-1] - patch[:, :, 1:]
    a2 = patch[:, :-1, :] - patch[:, 1:, :]
    return a1.abs().sum() + a2.abs().sum()
