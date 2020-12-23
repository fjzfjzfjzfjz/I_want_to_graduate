from torch import optim
from torch.utils import data as torch_data

from dataset import RelationDataset, TestRelationDataset
from metrics import *
from model import Model, AttributeModel
from uitls import *


def train_entity_embedding(model, train_set, test_set, id2URL, alpha, device, epoch=1000, batch_size=102400,
                           test_batch_size=64,
                           lr=1e-2, margin=1, norm=1, test_freq=50, dim=50, test_samples=5000, plot=False):
    # torch.random.manual_seed(1234)
    # model = Model(entity_count=len(id2URL), relation_count=len(id2URL), dim=dim,
    #                margin=margin, device=device, norm=norm, alpha=None)  # type:# torch.nn.Module
    # model = model.to(device)

    train_generator = RelationDataset(train_set)
    train_generator = torch.utils.data.DataLoader(train_generator, batch_size=batch_size)

    test_generator = TestRelationDataset(test_set, test_samples)
    test_generator = torch.utils.data.DataLoader(test_generator, batch_size=test_batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lls = []
    start_epoch_id = 1
    best_score = 0.0
    for epoch_id in range(start_epoch_id, epoch + 1):
        print(f"{train_entity_embedding.__name__} Starting epoch: ", epoch_id)
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
        # print(ll)
        lls.append(ll)
        if epoch_id % test_freq == 0:
            model.eval()
            print(f'{train_entity_embedding.__name__} loss', ll)
            print(f'{train_entity_embedding.__name__} test')
            hits_at_10, mrr = entity_embedding_test(model=model, data_generator=test_generator,
                                                    entities_count=model.entity_count,
                                                    device=device)

            print(f'{train_entity_embedding.__name__}', hits_at_10, mrr)
            score = hits_at_10
            if score > best_score:
                best_score = score
    return lls


def train_attr_embedding(model, train_set1, test_set1, train_set2, test_set2, id2URL1, id2URL2, attrs_only1,
                         attrs_only2,
                         alpha, attrs_rel_count, device, epoch=1000,
                         batch_size=2048, test_batch_size=64, use_lstm=True,
                         lr=1e-2, margin=1, norm=1, test_freq=50, dim=50, test_samples=5000, plot=False):
    # torch.random.manual_seed(1234)
    # a1 = torch.Tensor(attrs_only1).long().to(device)
    # a2 = torch.Tensor(attrs_only2).long().to(device)
    # alpha = torch.Tensor([alpha[i] for i in sorted(alpha.keys())]).float().to(device)

    # model = model.to(device)
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

    lls1 = []
    lls2 = []
    for epoch_id in range(start_epoch_id, epoch + 1):
        print(f"{train_attr_embedding.__name__} Starting epoch: ", epoch_id)
        model.train()
        print(f'{train_attr_embedding.__name__} train 1')
        model.train_first_set_a = True
        l1 = train_one_set(train_generator1, len(id2URL1), len(attrs_only1), optimizer1)
        print(f'{train_attr_embedding.__name__} train 2')
        model.train_first_set_a = False
        l2 = train_one_set(train_generator2, len(id2URL2), len(attrs_only2), optimizer2)
        print(f'{train_attr_embedding.__name__} loss', l1, l2)
        lls1.append(l1)
        lls2.append(l2)
        if epoch_id % test_freq == 0:
            model.eval()
            # print('loss', l1, l2)
            print(f'{train_attr_embedding.__name__} test')
            model.train_first_set_a = True
            hits_at_10, mrr = attr_embedding_test(model=model, data_generator=test_generator1,
                                                  entities_count=model.entity_count_a, attr_count=len(attrs_only1),
                                                  device=device, first=True)
            print(f'{train_attr_embedding.__name__} A {hits_at_10} {mrr}')
            model.train_first_set_a = False
            hits_at_10, mrr = attr_embedding_test(model=model, data_generator=test_generator2,
                                                  entities_count=model.entity_count_b, attr_count=len(attrs_only2),
                                                  device=device, first=False)
            print(f'{train_attr_embedding.__name__} B {hits_at_10} {mrr}')
            # score = hits_at_10
            # if score > best_score:
            #     best_score = score
    if plot:
        plot_loss(list(range(1, len(lls1) + 1)), lls1, lls2, title=train_attr_embedding.__name__)
    return lls1, lls2


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


def train(entity_train_set1, entity_test_set1, entity_train_set2, entity_test_set2, attr_train_set1, attr_test_set1,
          attr_train_set2, attr_test_set2, id2URL1, id2URL2, attrs_only1, attrs_only2,
          alpha, attrs_rel_count, device, KG2_entities, KG1_entities, epoch=1000, use_lstm=True,
          margin=1, norm=1, test_freq=50, dim=50):
    model1 = Model(entity_count=len(id2URL1), relation_count=len(id2URL1), dim=dim,
                   margin=margin, device=device, norm=norm, alpha=None)  # type:# torch.nn.Module
    model1 = model1.to(device)
    model2 = Model(entity_count=len(id2URL2), relation_count=len(id2URL2), dim=dim,
                   margin=margin, device=device, norm=norm, alpha=None)  # type:# torch.nn.Module
    model2 = model2.to(device)
    a1 = torch.Tensor(attrs_only1).long().to(device)
    a2 = torch.Tensor(attrs_only2).long().to(device)
    alpha = torch.Tensor([alpha[i] for i in sorted(alpha.keys())]).float().to(device)
    model3 = AttributeModel(entity_count_a=len(id2URL1), entity_count_b=len(id2URL2),
                            merged_rel_size=attrs_rel_count, use_lstm=use_lstm, alpha=alpha,
                            dim=dim, margin=margin, device=device, norm=norm, encoded_charset=encoded_charset,
                            attr_lookup_table_a=a1, attr_lookup_table_b=a2)  # type:# torch.nn.Module
    model3 = model3.to(device)
    optimizer = optim.Adam([model1.entities_emb, model2.entities_emb, model3.entities_emb_a, model3.entities_emb_b],
                           lr=1e-3)
    cc = torch.nn.CosineSimilarity(dim=1)

    # model1.load('model_checkpoint/train/model1')
    # model2.load('model_checkpoint/train/model2')
    # model3.load('model_checkpoint/train/model3')
    lls1, lls2, lls3, lls4, lls5 = [], [], [], [], []
    for i in range(1, epoch + 1):
        # 函数都返回loss列表
        print(f'Main epoch: {i}')
        # batch_size 409600
        ls1 = train_entity_embedding(model=model1, train_set=entity_train_set1, test_set=entity_test_set1,
                                     id2URL=id2URL1, alpha=None, lr=1e-2,
                                     device=device, batch_size=102400, test_batch_size=64, margin=margin, norm=norm,
                                     epoch=50, test_freq=50, dim=dim, test_samples=5000)
        ls2 = train_entity_embedding(model=model2, train_set=entity_train_set2, test_set=entity_test_set2,
                                     id2URL=id2URL2, alpha=None, lr=1e-2,
                                     device=device, batch_size=102400, test_batch_size=32, margin=margin, norm=norm,
                                     epoch=50, test_freq=50, dim=dim, test_samples=5000)
        # ls3, ls4 = train_attr_embedding(model3, attr_train_set1, attr_test_set1, attr_train_set2, attr_test_set2,
        #                                 id2URL1, id2URL2,
        #                                 attrs_only1, attrs_only2, alpha, attrs_rel_count,
        #                                 device, epoch=1, lr=1e-3, batch_size=1024, dim=dim, margin=margin,
        #                                 test_batch_size=16, test_samples=500, norm=norm,
        #                                 test_freq=50, use_lstm=use_lstm)
        # 前三个不会测试
        # joint learning
        # 直接用实体对齐
        # optimizer.zero_grad()
        # loss1 = 1 - cc(model1.entities_emb, model3.entities_emb_a)
        # loss2 = 1 - cc(model2.entities_emb, model3.entities_emb_b)
        # loss = loss1.sum() + loss2.sum()
        # loss.backward()
        # optimizer.step()

        # lls1.extend(ls1)
        # lls2.extend(ls2)
        # lls3.extend(ls3)
        # lls4.extend(ls4)
        # lls5.append(loss.item())
        # print('loss', ls1, ls2, ls3, ls4, loss.item())
        # model1.save('model_checkpoint/train/model1')
        # model2.save('model_checkpoint/train/model2')
        # model3.save('model_checkpoint/train/model3')
        print('loss', ls1, ls2)
        if i % test_freq == 0:
            # print('loss', ls1, ls2, ls3, ls4, loss.item())
            model1.eval()
            model2.eval()
            model3.eval()

            KG1_entities = torch.Tensor(KG1_entities).to(device).long()  # KG1的实体编号
            KG2_entities = torch.Tensor(KG2_entities).to(device).long()  # KG2相对应的对齐的实体编号
            true_label = torch.arange(0, KG2_entities.size(0)).to(device).long()
            # hits10, mr = alignment_hits10_and_mean_rank(model1.entities_emb[KG1_entities, :],
            #                                             model2.entities_emb[KG2_entities, :],
            #                                             true_label, device)
            hits10, mr = alignment_hits10_and_mean_rank(model1.entities_emb[KG1_entities, :],
                                                        model2.entities_emb,
                                                        KG2_entities, device)
            print(f'hits10,mr {hits10} , {mr}')
    print(lls1, lls2, lls3, lls4, lls5)
    plot_loss(list(range(1, epoch + 1)), lls1, lls2, lls3, lls4, lls5)


if __name__ == '__main__':
    from load_data import *

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # train_entity_embedding(train_tw, test_tw, id2URL_tw, None, device, batch_size=102400, test_freq=150, lr=0.1)
    # 转换顺序为一致
    aligned_fq_tw.sort(key=lambda x: x[1])
    aligned_items = [i[0] for i in aligned_fq_tw]
    indices = [i[1] for i in aligned_fq_tw]
    train(train_tw, test_tw, train_fq, test_fq, train_attr_tw, test_attr_tw, train_attr_fq, test_attr_fq, id2URL_tw,
          id2URL_fq, attrs_only_tw, attrs_only_fq, attr_alpha_tw_fq, merged_tw_fq_attr_rel_count, device, test_freq=1,
          use_lstm=True, epoch=1, KG2_entities=aligned_items, KG1_entities=indices)
    # train_attr_embedding(train_attr_tw, test_attr_tw, train_attr_fq, test_attr_fq,
    #                      id2URL_tw, id2URL_fq, attrs_only_tw, attrs_only_fq, None, None, merged_tw_fq_attr_rel_count,
    #                      device, lr=1e-3, batch_size=1024, test_batch_size=2, test_freq=1, use_lstm=True)
