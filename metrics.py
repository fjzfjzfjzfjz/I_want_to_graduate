from math import ceil

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


# sum 5120 lstm 2000
# TODO 是否需要取消替换尾部的测试用例？尾部如果全部随机的话，无法一次放入显存，考虑采样
def attr_hits10_and_mean_rank(model: torch.nn.Module, h: torch.LongTensor, r: torch.LongTensor,
                              t: torch.LongTensor, entity_count: int, attr_count: int, device, first=True,
                              batch_size=600):
    rank = 0
    hits10 = 0
    sample_size = h.size(0)
    # 一次性计算出结果，方法是，对每对h，r，组合出每一个t
    entity_ids = torch.arange(end=entity_count, device=device)
    attr_ids = torch.arange(end=attr_count, device=device)
    h = h.view(-1, 1)  # type: torch.LongTensor
    r = r.view(-1, 1)
    t = t.view(-1, 1)

    # batch_size = 256
    batches = ceil(entity_count / batch_size)
    # 替换头部h
    for batch in range(batches):
        start = batch * batch_size
        end = (batch + 1) * batch_size
        this_entity_ids = entity_ids[start:end]
        entities = this_entity_ids.view(-1, 1).repeat(sample_size, 1)
        relations = r.repeat(1, this_entity_ids.size(0)).view(-1, 1)
        tails = t.repeat(1, this_entity_ids.size(0)).view(-1, 1)
        replaced_heads = torch.cat([entities, relations, tails], dim=1)
        scores = model.predict(replaced_heads, first).view(sample_size, -1)
        true_label = h
        # hits 10
        zero_tensor = torch.tensor([0], device=device)
        one_tensor = torch.tensor([1], device=device)
        _, indices = scores.topk(k=10, largest=False, dim=1)  # 自动安装1维度排序
        hits10 += torch.where(indices == true_label, one_tensor, zero_tensor).sum().item()  # 自动广播
        # mean rank
        indices = scores.argsort()  # 直接返回indices的sort
        rank += torch.nonzero(indices == true_label)[:, 1].float().sum().item()

    # 替换尾部
    batches = ceil(attr_count / batch_size)
    for batch in range(batches):
        start = batch * batch_size
        end = (batch + 1) * batch_size
        this_attr_ids = attr_ids[start:end]
        relations = r.repeat(1, this_attr_ids.size(0)).view(-1, 1)
        heads = h.repeat(1, this_attr_ids.size(0)).view(-1, 1)
        attrs = this_attr_ids.view(-1, 1).repeat(sample_size, 1)
        replaced_tails = torch.cat([heads, relations, attrs], dim=1)

        scores = model.predict(replaced_tails, first).view(sample_size, -1)
        true_label = t
        # hits 10
        zero_tensor = torch.tensor([0], device=device)
        one_tensor = torch.tensor([1], device=device)
        _, indices = scores.topk(k=10, largest=False, dim=1)  # 自动安装1维度排序
        hits10 += torch.where(indices == true_label, one_tensor, zero_tensor).sum().item()  # 自动广播
        # mean rank
        indices = scores.argsort()  # 直接返回indices的sort
        rank += torch.nonzero(indices == true_label)[:, 1].float().sum().item()

    return hits10, rank


# TODO 可能是代码编写错误导致效果不好
# TODO true label有问题，true label是传入的之前的index，而不是entity_emb2的index
# batch_size 256
def alignment_hits10_and_mean_rank(entity_emb1, entity_emb2, true_label, device, batch_size=32):
    sim = torch.nn.CosineSimilarity(dim=1)
    entity_ids1 = torch.arange(end=entity_emb1.size(0), device=device)
    entity_ids2 = torch.arange(end=entity_emb2.size(0), device=device)
    rank = 0
    hits10 = 0

    # 每次只处理一部分entity id，即正常的batch操作
    batches = ceil(entity_emb1.size(0) / batch_size)
    for batch in range(batches):
        start = batch * batch_size
        end = (batch + 1) * batch_size
        this_entity_emb1 = entity_emb1[start:end, :]
        this_entity_ids1 = entity_ids1[start:end]
        this_true_label = true_label[start:end].view(-1, 1)

        entities1 = this_entity_ids1.view(-1, 1).repeat(1, entity_emb2.size(0)).view(-1)
        entities2 = entity_ids2.view(-1, 1).repeat(this_entity_emb1.size(0), 1).view(-1)  # 变成普通向量
        entities1_cos_input = entity_emb1[entities1, :]  # 这是对的
        entities2_cos_input = entity_emb2[entities2, :]
        # 一个1对应每个2的
        similarity = sim(entities1_cos_input, entities2_cos_input).view(-1, entity_emb2.size(0))

        # hits 10
        zero_tensor = torch.tensor([0], device=device)
        one_tensor = torch.tensor([1], device=device)
        _, indices = similarity.topk(k=10, largest=False, dim=1)  # 自动安装1维度排序
        hits10 += torch.where(indices == this_true_label, one_tensor, zero_tensor).sum().item()
        # mean rank
        indices = similarity.argsort()  # 直接返回indices的sort
        rank += torch.nonzero(indices == this_true_label)[:, 1].float().sum().item()
    print(hits10, rank)
    return hits10 / entity_emb1.size(0) / 2 * 100, rank / entity_emb1.size(0) / 2


def acc(pred, label):
    pass


if __name__ == '__main__':
    from load_data import *

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pos_triples = torch.Tensor(rel_tw).to(device).long()
    structure_hits10_and_mean_rank(None, pos_triples, len(id2URL_tw), device)
