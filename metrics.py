from distance import *


def structure_hits10_and_mean_rank(model: torch.nn.Module, h: torch.LongTensor, r: torch.LongTensor,
                                   t: torch.LongTensor, entity_count: int, device):
    sample_size = h.size(0)
    # 一次性计算出结果，方法是，对每对h，r，组合出每一个t
    entity_ids = torch.arange(end=entity_count, device=device)
    h = h.view(-1, 1)  # type: torch.LongTensor
    r = r.view(-1, 1)
    t = t.view(-1, 1)

    heads = h.repeat(1, entity_count).view(-1, 1)  # 实现111222形式，利用到了view的特性，会按行列顺序消除维度
    relations = r.repeat(1, entity_count).view(-1, 1)
    tails = t.repeat(1, entity_count).view(-1, 1)
    entities = entity_ids.view(-1, 1).repeat(sample_size, 1)
    replaced_heads = torch.cat([entities, relations, tails], dim=1)
    replaced_tails = torch.cat([heads, relations, entities], dim=1)
    test_set = torch.cat([replaced_heads, replaced_tails], dim=0)
    scores = model.predict(test_set).view(-1, entity_count)
    true_label = torch.cat([h, t], dim=0)
    # hits 10
    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)
    _, indices = scores.topk(k=10, largest=False, dim=1)  # 自动安装1维度排序
    hits10 = torch.where(indices == true_label, one_tensor, zero_tensor).sum().item()
    # mean rank
    indices = scores.argsort()  # 直接返回indices的sort
    mr = torch.nonzero(indices == true_label)[:, 1].float().sum().item()
    return hits10, mr


# TODO 是否需要取消替换尾部的测试用例？尾部如果全部随机的话，无法一次放入显存，考虑采样
def attr_hits10_and_mean_rank(model: torch.nn.Module, h: torch.LongTensor, r: torch.LongTensor,
                              t: torch.LongTensor, entity_count: int, attr_count: int, device, first=True):
    sample_size = h.size(0)
    # 一次性计算出结果，方法是，对每对h，r，组合出每一个t
    entity_ids = torch.arange(end=entity_count, device=device)
    attr_ids = torch.arange(end=attr_count, device=device)
    # 采样，否则太多了，。。,双采样
    attr_samples = entity_count
    entity_samples = entity_count
    idxs = torch.randint(0, attr_count, (attr_samples,))
    attr_ids = attr_ids[idxs]
    idxs = torch.randint(0, entity_count, (entity_samples,))
    entity_ids = entity_ids[idxs]

    h = h.view(-1, 1)  # type: torch.LongTensor
    r = r.view(-1, 1)
    t = t.view(-1, 1)

    entities = entity_ids.view(-1, 1).repeat(sample_size, 1)
    attrs = attr_ids.view(-1, 1).repeat(sample_size, 1)

    # TODO 错误
    relations = r.repeat(1, entity_samples).view(-1, 1)
    tails = t.repeat(1, entity_samples).view(-1, 1)
    replaced_heads = torch.cat([entities, relations, tails], dim=1)

    relations = r.repeat(1, attr_samples).view(-1, 1)
    heads = h.repeat(1, attr_samples).view(-1, 1)
    replaced_tails = torch.cat([heads, relations, attrs], dim=1)

    test_set = torch.cat([replaced_heads, replaced_tails], dim=0)
    scores = model.predict(test_set, first).view(-1, entity_samples)
    true_label = torch.cat([h, t], dim=0)
    # hits 10
    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)
    _, indices = scores.topk(k=10, largest=False, dim=1)  # 自动安装1维度排序
    hits10 = torch.where(indices == true_label, one_tensor, zero_tensor).sum().item()  # 自动广播
    # mean rank
    indices = scores.argsort()  # 直接返回indices的sort
    rank = torch.nonzero(indices == true_label)[:, 1].float().sum().item()
    return hits10, rank


def acc(pred, label):
    pass


if __name__ == '__main__':
    from load_data import *

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pos_triples = torch.Tensor(rel_tw).to(device).long()
    structure_hits10_and_mean_rank(None, pos_triples, len(id2URL_tw), device)
