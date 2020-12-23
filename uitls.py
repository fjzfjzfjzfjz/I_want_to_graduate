# 按顺序返回值
import matplotlib.pyplot as plt
import seaborn
import torch


def get_structure_corrupted(triples: torch.Tensor, entity_or_attr_count: int, device: torch.device):
    """
    使用api 加速
    :param triples: rel
    :param entity_or_attr_count: 实体总数，或者属性attr总数
    :return: 替换成这个关系不具备的其他的;随机替换头尾;只返回编号
    """
    # 忽略重复，因为很少（几十万个关系出现2 3千重复），每个样例随机选择头尾
    h = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]
    head_or_tail = torch.randint(high=2, size=(h.size(0),), device=device)
    # head_or_tail = head_or_tail.view(-1, 1).repeat(1, h.size(1))  # 行重复一次，列重复n次
    # head_or_tail = torch.stack([head_or_tail] * h.size(1), dim=1)
    random_entities = torch.randint(high=entity_or_attr_count, size=(h.size(0),), device=device)
    # random_entities = entity_embedding[random_entities, :]
    broken_heads = torch.where(head_or_tail == 1, random_entities, h)
    broken_tails = torch.where(head_or_tail == 0, random_entities, t)
    return torch.stack([broken_heads, r, broken_tails], dim=1)


def get_attr_corrupted(triples: torch.Tensor, entity_count: int, attr_count: int, device: torch.device):
    h = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]
    head_or_tail = torch.randint(high=2, size=(h.size(0),), device=device)
    random_entities = torch.randint(high=entity_count, size=(h.size(0),), device=device)
    random_attrs = torch.randint(high=attr_count, size=(h.size(0),), device=device)
    broken_heads = torch.where(head_or_tail == 1, random_entities, h)
    broken_tails = torch.where(head_or_tail == 0, random_attrs, t)
    return torch.stack([broken_heads, r, broken_tails], dim=1)


def plot_loss(ep, *ls, **kwargs):
    """

    :param ep: 对应的epoch编号
    :param ls: many array of losses
    """
    seaborn.set_style("ticks")
    title = kwargs.get('title', 'title')
    for i in ls:
        seaborn.lineplot(ep, i, legend='full')
        plt.title(title)
        plt.show()


if __name__ == '__main__':
    h = torch.Tensor([[1, 2], [3, 4]])
    r = torch.Tensor([[1, 2]])
    t = torch.Tensor([[1, 2], [3, 4]])
    e = torch.Tensor([[1, 1], [2, 2], [3, 3]])
    # print(get_structure_corrupted(h, r, t, e, rel_tw_cor))
    plot_loss([1, 2], [1, 1], [2, 2])
