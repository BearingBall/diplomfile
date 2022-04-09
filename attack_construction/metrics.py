import numpy as np
import torch


def general_objectness(labels, device):  # as in InvisibleCloak
    score = torch.tensor(0.0).to(device)
    for i in range(len(labels["labels"])):
        # and labels["scores"][i] > 0.6
        if labels["labels"][i] == 1:
            score += max(labels["scores"][i] + torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))**2
    return score


# TV - total variation penalty (smooth for patch)
def total_variation(patch):
    # wiki: https://en.wikipedia.org/wiki/Total_variation_denoising
    a1 = patch[:, :, :-1] - patch[:, :, 1:]
    a2 = patch[:, :-1, :] - patch[:, 1:, :]
    return a1.abs().sum() + a2.abs().sum()

printability_colors = [
[0.10588, 0.054902, 0.1098],
[0.48235, 0.094118, 0.16863],
[0.50196, 0.52549, 0.17647],
[0.082353, 0.31765, 0.18431],
[0.47843, 0.61176, 0.51765],
[0.07451, 0.3098, 0.45882],
[0.67843, 0.14902, 0.18039],
[0.086275, 0.14118, 0.26275],
[0.26667, 0.36863, 0.47843],
[0.76078, 0.54118, 0.5451],
[0.73333, 0.49412, 0.27451],
[0.25882, 0.35294, 0.18039],
[0.47843, 0.22353, 0.36471],
[0.27059,0.086275, 0.11765],
[0.7098,0.32157, 0.2],
[0.27451, 0.13725, 0.29412],
[0.75294, 0.75686, 0.63137],
[0.28627, 0.54902, 0.41176],
[0.47451, 0.2902, 0.15294],
[0.74902, 0.70196, 0.28627],
[0.098039, 0.42745, 0.44314],
[0.50588, 0.65098, 0.65882],
[0.12549, 0.42745, 0.23529],
[0.4902, 0.58431, 0.33725],
[0.26275, 0.49412, 0.26275],
[0.07451, 0.14902, 0.12549],
[0.090196, 0.20392, 0.36078],
[0.68627, 0.15686, 0.30196],
[0.30196, 0.5451, 0.57647],
[0.71765, 0.32941, 0.40784]]


def printability_loss(patch):
    patch = patch.permute(1, 2, 0)
    dist = []
    for i in range(patch.size()[0]):
        for j in range(patch.size()[1]):
            dist.append(((patch[i][j] - torch.tensor(printability_colors)).square()).sum(1).min())

    return torch.mean(torch.tensor(dist))