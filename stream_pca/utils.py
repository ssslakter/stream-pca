import torch


def get_device(use_cuda=True):
    return torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
