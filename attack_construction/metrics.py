import numpy as np
import torch


def general_objectness(labels):  # as in InvisibleCloak
    score = torch.tensor(0.0).cuda()
    for i in range(len(labels["labels"])):
        # and labels["scores"][i] > 0.6
        if labels["labels"][i] == 1:
            score += max(labels["scores"][i] + torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())**2
    return score


# TV - total variation penalty (smooth for patch)
def total_variation(patch):
    # wiki: https://en.wikipedia.org/wiki/Total_variation_denoising
    a1 = patch[:, :, :-1] - patch[:, :, 1:]
    a2 = patch[:, :-1, :] - patch[:, 1:, :]
    return a1.abs().sum() + a2.abs().sum()


def get_printable_colors(patch_size):
    printability_list = []

    with open('printable_colors.txt') as f:
        for line in f:
            printability_list.append(line.split(","))

    printability_array = []
    for printability_triplet in printability_list:
        printability_imgs = []
        red, green, blue = printability_triplet
        printability_imgs.append(np.full((patch_size, patch_size), red))
        printability_imgs.append(np.full((patch_size, patch_size), green))
        printability_imgs.append(np.full((patch_size, patch_size), blue))
        printability_array.append(printability_imgs)

    printability_array = np.asarray(printability_array)
    printability_array = np.float32(printability_array)
    pa = torch.from_numpy(printability_array)
    return pa


def nps_score(printability_array, patch):
    # calculate euclidian distance between colors in patch and colors in printability_array 
    # square root of sum of squared difference
    color_dist = (patch - printability_array+0.000001)
    color_dist = color_dist ** 2
    color_dist = torch.sum(color_dist, 1)+0.000001
    color_dist = torch.sqrt(color_dist)
    # only work with the min distance
    color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
    # calculate the nps by summing over all pixels
    nps_score = torch.sum(color_dist_prod,0)
    nps_score = torch.sum(nps_score,0)
    return nps_score/torch.numel(patch)
