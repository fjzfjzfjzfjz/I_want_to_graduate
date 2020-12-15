# 按顺序返回值
import matplotlib.pyplot as plt
import seaborn
import torch

from load_data import *


def get_structure_corrupted(h, r, t, entity_embedding: torch.Tensor, device):
    """
    使用api 加速
    :param entity_embedding: 某个KG的关注关系集合,[a,b]
    :param h:
    :param r:只有一个关系，一行
    :param t:
    :return: 替换成这个关系不具备的其他的;随机替换头尾
    """
    head = np.random.random() < 0.5
    # print('head ', head)
    if head:
        idxs = np.random.randint(0, entity_embedding.shape[0], (h.shape[0],), dtype=np.int64)
        ch = torch.index_select(entity_embedding, 0, torch.from_numpy(idxs).to(device))
        return ch, r, t
    else:
        idxs = np.random.randint(0, entity_embedding.shape[0], (h.shape[0],), dtype=np.int64)
        ct = torch.index_select(entity_embedding, 0, torch.from_numpy(idxs).to(device))
        return h, r, ct
    # 产生随机数本身不慢，可以用for
    # if head:
    #     cc = []
    #     for i in range(h.shape[0]):
    #         tt = np.random.randint(0, entity_embedding.shape[0], dtype=np.int64)
    #         tt_embedding=torch.index_select(entity_embedding, 0, torch.from_numpy(tt).to(device))
    #         while True:
    #             if t not in rel_dict or (t in rel_dict ):break
    #             if tt_embedding
    #     idxs = np.random.randint(0, entity_embedding.shape[0], (h.shape[0],), dtype=np.int64)
    #     ch = torch.index_select(entity_embedding, 0, torch.from_numpy(idxs).to(device))
    #     return ch, r, t
    # else:
    #     idxs = np.random.randint(0, entity_embedding.shape[0], (h.shape[0],), dtype=np.int64)
    #     ct = torch.index_select(entity_embedding, 0, torch.from_numpy(idxs).to(device))
    #     return h, r, ct

    # for i in range(d.shape[0]):
    #     if i % 10000 == 0: print(i)
    #     cc = np.random.choice(idxs)
    # while cc == i:
    #     cc = np.random.choice(idxs)
    # TODO 因随机到重复的概率很低，所以不考虑重复
    # TODO 反例不应h+r=t，应该排除，否则会难以收敛
    # d[i] = entity_embedding[cc]
    # return h, r, torch.Tensor([1, 2, 3])  # 作为桩模块
    # if head:
    #     return d, r, t
    # else:
    #     return h, r, d


def get_attr_corrupted(h, r, t, entity_embedding: torch.Tensor, char_embedding: torch.Tensor
                       , device, f, cor_rel: dict):
    head = cor_rel['head']
    rel = np.array(cor_rel['col'])


def plot_loss(ls, ep):
    """

    :param ep: 对应的epoch编号
    :param ls: an array of losses
    """
    seaborn.set_style("ticks")
    seaborn.lineplot(ep, ls, legend='full')
    plt.show()


if __name__ == '__main__':
    h = torch.Tensor([[1, 2], [3, 4]])
    r = torch.Tensor([[1, 2]])
    t = torch.Tensor([[1, 2], [3, 4]])
    e = torch.Tensor([[1, 1], [2, 2], [3, 3]])
    print(get_structure_corrupted(h, r, t, e, rel_tw_cor))
