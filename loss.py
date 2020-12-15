import numpy as np
import torch

from distance import l1_norm
from uitls import get_structure_corrupted


def structure_loss(h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, cor_x=None, dis=l1_norm, gamma=1):
    # 只有关注关系。。
    """
    :param t:所有的头矩阵
    :param r:矩阵，一行，代表关注
    :param h:所有的尾，矩阵
    :param cor_x:额外的反例，可以不提供
    :return: loss
    """
    rt = torch.zeros(1)
    for i in r:
        # 每一个关系
        # 会自动广播，导致维度一致
        ch, ci, ct = get_structure_corrupted(h, r, t, None)
        rt += torch.max(dis(h + i - t) - dis(ch + ci - ct) + gamma, torch.Tensor([0]))
    return rt


if __name__ == '__main__':
    h = torch.from_numpy(np.array([[1, -1, 2], [1, -1, 2]]))
    r = torch.from_numpy(np.array([1, 0, 2]))
    t = torch.from_numpy(np.array([[1, -1, 2], [1, -1, 2]]))
    print(structure_loss(h, r, t))
