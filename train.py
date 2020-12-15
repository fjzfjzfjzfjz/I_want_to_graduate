from metrics import *
from model import Model, AttrModel
from uitls import *


def train_tw_entity(epoch=1000, lr=1e-3, dis=l1_norm, gamma=1, dim=75):
    rel_triples = torch.tensor(rel_tw, dtype=torch.int32, device=device)
    mxx = max([int(i) for i in id2URL_tw]) + 1
    model = Model(input_samples_dim=mxx, dis=dis, features_dim=dim, gamma=gamma).to(device)  # 因为编号不是从0开始，所以要有空闲部分
    # model.to(device)会自动把所有的self。xxxtensor变成.to(device)

    # model.load('model_checkpoint/test/')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ls = []
    for i in range(epoch):
        optimizer.zero_grad()
        loss = model(rel_triples, rel_tw_cor)
        # print(model.rel_embeddings)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('epoch %d ,loss %.2f' % (i + 1, loss.item()))
        ls.append(loss.item())
        if (i + 1) % 500 == 0:
            hits10, mr = structure_hits10_and_mean_rank(rel_triples, model.entity_embeddings,
                                                        model.rel_embeddings, dis, device)
            print(f'{hits10 * 100}% {mr}')
    plot_loss(ls, range(1, epoch + 1))
    model.save('model_checkpoint/test/')


def train_attr(epoch=10, lr=1e-3, dis=l1_norm):
    mxx = max([int(i) for i in id2URL_tw]) + 1
    model = Model(input_samples_dim=mxx, dis=dis).to(device)
    batch = 100
    sep = len(attr_rel_tw) // batch
    model1 = AttrModel().to(device)
    optimizer = torch.optim.Adam([model.entity_embeddings, model1.rel_attr_embeddings, model1.char_embeddings], lr=lr)
    ls = []
    for i in range(epoch):
        for j in range(batch):
            data = attr_rel_tw[j * sep:(j + 1) * sep]
            optimizer.zero_grad()
            loss = model1(data, model.entity_embeddings)
            print(loss)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            if (j + 1) % 10 == 0:
                print(sum(ls))
            ls.append(loss.item())
            # if (i + 1) % 100 == 0:
            #     hits10, mr = structure_hits10_and_mean_rank(rel_triples, model.entity_embeddings,
            #                                             model.rel_embeddings, dis, device)
            # print(f'{hits10 * 100}% {mr}')
        if (i + 1) % 1 == 0:
            print('epoch %d ,loss %.2f' % (i + 1, sum(ls)))
    plot_loss(ls, range(1, epoch + 1))
    model.save('model_checkpoint/test/')


def train_attr_with_sum(epoch=100, lr=1e-2, dis=l1_norm):
    mxx = max([int(i) for i in id2URL_tw]) + 1
    model = Model(input_samples_dim=mxx, dis=dis).to(device)
    model1 = AttrModel().to(device)
    optimizer = torch.optim.Adam([model.entity_embeddings, model1.rel_attr_embeddings, model1.char_embeddings], lr=lr)
    ls = []
    for i in range(epoch):
        optimizer.zero_grad()
        loss = model1(attr_rel_tw, model.entity_embeddings)
        print(loss.item())
        loss.backward()
        optimizer.step()
        ls.append(loss.item())
        if (i + 1) % 10 == 0:
            print('epoch %d ,loss %.2f' % (i + 1, sum(ls)))
    plot_loss(ls, range(1, epoch + 1))
    model.save('model_checkpoint/test/attr')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_tw_entity()
    # train_attr_with_sum()
