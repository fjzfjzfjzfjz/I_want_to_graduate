from torch import optim
from torch.utils import data as torch_data

from TransEtest import *
from dataset import RelationDataset, TestRelationDataset
from metrics import *
from model import Model, AttributeModel
from uitls import *


def train_entity_embedding(train_set, test_set, id2URL, alpha, device, epoch=1000, batch_size=102400,
                           test_batch_size=64,
                           lr=1e-2, margin=1, norm=1, test_freq=50, dim=50, test_samples=5000):
    # torch.random.manual_seed(1234)

    train_generator = RelationDataset(train_set)
    train_generator = torch.utils.data.DataLoader(train_generator, batch_size=batch_size)

    test_generator = TestRelationDataset(test_set, test_samples)
    test_generator = torch.utils.data.DataLoader(test_generator, batch_size=test_batch_size)
    model = Model(entity_count=len(id2URL), relation_count=len(id2URL), dim=dim,
                  margin=margin, device=device, norm=norm, alpha=alpha)  # type:# torch.nn.Module
    # model = TransE(entity_count=len(id2URL), relation_count=len(id2URL), dim=dim,
    #                margin=margin,
    #                device=device, norm=norm)  # type:torch.nn.Module
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch_id = 1
    best_score = 0.0
    for epoch_id in range(start_epoch_id, epoch + 1):
        print("Starting epoch: ", epoch_id)
        model.train()
        ll = 0
        for local_heads, local_relations, local_tails in train_generator:
            local_heads, local_relations, local_tails = (local_heads.to(device), local_relations.to(device),
                                                         local_tails.to(device))
            positive_triples = torch.stack((local_heads, local_relations, local_tails), dim=1)
            negative_triples = get_structure_corrupted(positive_triples, model.entity_count, device)

            optimizer.zero_grad()
            loss = model(positive_triples, negative_triples)
            loss.mean().backward()

            loss = loss.data.cpu()
            ll += loss.sum().item()
            optimizer.step()
        print(ll)
        if epoch_id % test_freq == 0:
            model.eval()
            print('loss', ll)
            print('test')
            hits_at_10, mrr = entity_embedding_test(model=model, data_generator=test_generator,
                                                    entities_count=model.entity_count,
                                                    device=device)

            print(hits_at_10, mrr)
            score = hits_at_10
            if score > best_score:
                best_score = score


# TODO 两个一起,可以先训练1，在训练2，这样就不用区分实体嵌入了；
# TODO 使用where！！
def train_attr_embedding(train_set1, test_set1, train_set2, test_set2, id2URL1, id2URL2, attrs_only1, attrs_only2,
                         alpha1, alpha2, attrs_rel_count, device, epoch=1000,
                         batch_size=2048, test_batch_size=64, use_lstm=True,
                         lr=1e-2, margin=1, norm=1, test_freq=50, dim=50, test_samples=5000):
    # torch.random.manual_seed(1234)
    a1 = torch.Tensor(attrs_only1).long().to(device)
    a2 = torch.Tensor(attrs_only2).long().to(device)
    model = AttributeModel(entity_count_a=len(id2URL1), entity_count_b=len(id2URL2),
                           merged_rel_size=attrs_rel_count, use_lstm=use_lstm,
                           dim=dim, margin=margin, device=device, norm=norm, encoded_charset=encoded_charset,
                           attr_lookup_table_a=a1, attr_lookup_table_b=a2)  # type:# torch.nn.Module
    model = model.to(device)
    start_epoch_id = 1
    step = 0
    best_score = 0.0
    attrs_count = len(attrs_only1) + len(attrs_only2)

    # optimizer1 = optim.Adam([model.char_embeddings, model.rel_embeddings, model.entities_emb_a], lr=lr)
    # optimizer2 = optim.Adam([model.char_embeddings, model.rel_embeddings, model.entities_emb_b], lr=lr)
    optimizer1 = optim.Adam(model.parameters(), lr=lr)
    optimizer2 = optim.Adam(model.parameters(), lr=lr)
    train_generator1 = RelationDataset(train_set1)
    train_generator1 = torch.utils.data.DataLoader(train_generator1, batch_size=batch_size)
    test_generator1 = TestRelationDataset(test_set1, test_samples)
    test_generator1 = torch.utils.data.DataLoader(test_generator1, batch_size=test_batch_size)

    train_generator2 = RelationDataset(train_set2)
    train_generator2 = torch.utils.data.DataLoader(train_generator2, batch_size=batch_size)
    test_generator2 = TestRelationDataset(test_set2, test_samples)
    test_generator2 = torch.utils.data.DataLoader(test_generator2, batch_size=test_batch_size)

    def train_one_set(train_generator, entity_count, attr_count, optimizer):
        model.train()
        ll = 0
        for local_heads, local_relations, local_tails in train_generator:
            local_heads, local_relations, local_tails = (local_heads.to(device), local_relations.to(device),
                                                         local_tails.to(device))
            positive_triples = torch.stack((local_heads, local_relations, local_tails), dim=1)
            negative_triples = get_attr_corrupted(positive_triples, entity_count, attr_count, device)

            optimizer.zero_grad()
            loss = model(positive_triples, negative_triples)
            loss.mean().backward()

            loss = loss.data.cpu()
            ll += loss.sum().item()
            optimizer.step()
        return ll

    for epoch_id in range(start_epoch_id, epoch + 1):
        print("Starting epoch: ", epoch_id)
        model.train()
        print('train 1')
        model.train_first_set_a = True
        l1 = train_one_set(train_generator1, len(id2URL1), len(attrs_only1), optimizer1)
        print('train 2')
        model.train_first_set_a = False
        l2 = train_one_set(train_generator2, len(id2URL2), len(attrs_only2), optimizer2)
        print('loss', l1, l2)
        if epoch_id % test_freq == 0:
            model.eval()
            # print('loss', l1, l2)
            print('test')
            model.train_first_set_a = True
            hits_at_10, mrr = attr_embedding_test(model=model, data_generator=test_generator1,
                                                  entities_count=model.entity_count_a, attr_count=len(attrs_only1),
                                                  device=device, first=True)
            print(f'A {hits_at_10} {mrr}')
            model.train_first_set_a = False
            hits_at_10, mrr = attr_embedding_test(model=model, data_generator=test_generator2,
                                                  entities_count=model.entity_count_b, attr_count=len(attrs_only2),
                                                  device=device, first=False)
            print(f'B {hits_at_10} {mrr}')
            # score = hits_at_10
            # if score > best_score:
            #     best_score = score


def entity_embedding_test(model, data_generator, entities_count, device):
    examples_count = 0
    hits10 = 0
    mr = 0

    # 去除无用显存
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    for head, relation, tail in data_generator:
        head, relation, tail = head.to(device), relation.to(device), tail.to(device)
        t1, t2 = structure_hits10_and_mean_rank(model, head, relation, tail, entities_count, device)
        hits10 += t1
        mr += t2
        examples_count += head.size(0)
    hits_at_10_score = hits10 / examples_count / 2 * 100
    mr_score = mr / examples_count / 2
    return hits_at_10_score, mr_score


def attr_embedding_test(model, data_generator, entities_count, attr_count, device, first=True):
    examples_count = 0
    hits10 = 0
    rank = 0

    # 去除无用显存
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    for head, relation, tail in data_generator:
        head, relation, tail = head.to(device), relation.to(device), tail.to(device)
        t1, t2 = attr_hits10_and_mean_rank(model, head, relation, tail, entities_count, attr_count, device, first)
        hits10 += t1
        rank += t2
        examples_count += head.size(0)
    hits_at_10_score = hits10 / examples_count / 2 * 100
    mr_score = rank / examples_count / 2
    return hits_at_10_score, mr_score
    # return hits10, rank


if __name__ == '__main__':
    from load_data import *

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # train_entity_embedding(train_tw, test_tw, id2URL_tw, alpha_tw, device, batch_size=102400, test_freq=150, lr=0.1)
    train_attr_embedding(train_attr_tw, test_attr_tw, train_attr_fq, test_attr_fq,
                         id2URL_tw, id2URL_fq, attrs_only_tw, attrs_only_fq, None, None, merged_tw_fq_attr_rel_count,
                         device, lr=1e-3, batch_size=1024, test_batch_size=2, test_freq=1, use_lstm=True)
    # train_tw_entity()
    # train_tw_entity()
    # train_attr_with_sum()
