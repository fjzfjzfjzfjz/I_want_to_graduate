import numpy as np
import torch


# 都应返回 a x 1,表示每一组h r t关系的距离
def l1_norm(x):
    return torch.norm(x, dim=1)


def l2_norm(x, sqrt=True):
    t = torch.sum(torch.square(x), dim=1, dtype=torch.float64)
    if sqrt:
        return torch.sqrt(t)
    else:
        return t


if __name__ == '__main__':
    a = torch.from_numpy(np.array([1, -1, 2]))
    print(l2_norm(a))
