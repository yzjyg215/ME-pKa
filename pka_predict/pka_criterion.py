import torch


def criterion2(y_pre, y_ori):
    return ((y_ori - y_pre)**2).mean()


def criterion(y_pre, y_ori):
    return torch.abs(y_ori - y_pre).mean()


def criterion3(y_pre, y_ori):
    loss1 = torch.abs(y_ori - y_pre)
    loss2 = (y_ori - y_pre)**2 * 2
    return torch.maximum(loss1, loss2).mean()

