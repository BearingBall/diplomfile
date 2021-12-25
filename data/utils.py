import torch


def image_to_tensor(image):
    return torch.tensor(image).permute(2, 0, 1).float() / 255


def tensor_to_image(tensor):
    return tensor.permute(1, 2, 0).detach().cpu().numpy() * 255
