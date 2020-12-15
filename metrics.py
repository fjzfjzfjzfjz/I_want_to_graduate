import numpy as np

from distance import *


def structure_hits10_and_mean_rank(test_triples: torch.Tensor, entity_embeddings: torch.Tensor, r: torch.Tensor,
                                   dis, device, sample_size=5000):
    # 替换头尾,选择头部
    # 无法使用三维张量来进行计算，因为矩阵太大了
    hits10 = 0
    rank = 0
    ll = sample_size
    # 使用20%测试数据
    ii = 0
    # 使用采样
    idxs = np.random.randint(0, test_triples.shape[0], (sample_size,), dtype=np.int64)
    tt = torch.index_select(test_triples, 0, torch.from_numpy(idxs).to(device))
    # for triple in tt:
    #     h = entity_embeddings[int(triple[0])]
    #     score = dis(h + r - entity_embeddings)
    #     _, indices = torch.sort(score)
    #     # 从indices找triple[1]是第几个
    #     _, indices = torch.sort(indices)
    #     # print(indices[triple[1]])
    #     if indices[triple[1]].item() < 10:
    #         hits10 += 1
    #     rank += indices[triple[1]].item()
    #     ii += 1
    #     if ii % 1000 == 0:
    #         print(ii, ll, hits10 / ii)
    #
    #     # 替换头

    return hits10 / sample_size, rank / sample_size


def acc(pred, label):
    pass


if __name__ == '__main__':
    pass
