import os

import torch.nn as nn
import torch.nn.init as init

from dataset import *
from distance import *
from uitls import *

best_gamma = [1, 5, 10]
best_dim = [50, 75, 100, 200]
best_learning_rate = [0.001, 0.01, 0.1]


# TODO 不要使用160维度，显存不足

class Model(nn.Module):
    def __init__(self, entity_count: int, relation_count: int, device, alpha: dict, norm=1, dim=50, margin=1.0):
        super(Model, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.entities_emb = nn.Parameter(torch.Tensor(entity_count, dim))
        self.rel_embeddings = nn.Parameter(torch.Tensor(relation_count, dim))
        self.norm = norm
        self.entity_count = entity_count
        self.features_dim = dim
        self.margin = margin
        # 按照关系id生成tensor
        self.alpha = torch.Tensor([alpha[i] for i in range(relation_count)]).float().to(device).view(-1, 1)
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

        self.reset_parameters()

    def predict(self, triplets: torch.LongTensor):
        return self._distance(triplets)

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        with torch.no_grad():
            self.entities_emb.data = self.entities_emb.data / torch.norm(self.entities_emb, p=2, dim=1).view(-1, 1)

        pd = self._distance(positive_triplets)
        nd = self._distance(negative_triplets)
        # r = positive_triplets[:, 1]
        # factor = self.alpha[r, 0]
        # rt = torch.max(factor * (pd - nd) + self.margin, torch.zeros(1, device=self.device))
        rt = torch.max(pd - nd + self.margin, torch.zeros(1, device=self.device))
        # target = torch.tensor([-1], dtype=torch.long, device=self.device)
        # rt = self.criterion(self.dis(h + r - t), self.dis(ch + cr - ct), target)
        return rt
        # 不是一个值

    def reset_parameters(self):
        uniform_range = 6 / np.sqrt(self.features_dim)
        init.uniform_(self.entities_emb, -uniform_range, uniform_range)
        init.uniform_(self.rel_embeddings, -uniform_range, uniform_range)
        self.rel_embeddings.data = self.rel_embeddings / torch.norm(self.rel_embeddings, p=1, dim=1).view(-1, 1)

    def _distance(self, triples: torch.LongTensor):
        assert triples.size(1) == 3
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]
        return (self.entities_emb[heads, :] + self.rel_embeddings[relations, :]
                - self.entities_emb[tails, :]).norm(p=self.norm, dim=1)

    def save(self, path):
        """
        存储所有
        :param path:
        :return:
        """
        # Module对象显式定义的Parameter类型的属性会放到self._parameters字典中,是有序字典
        if not os.path.exists(path):
            os.mkdir(path)
        for e in self._parameters:
            torch.save(self._parameters[e], path + f'/{e}.torch_save')

    def load(self, path):
        """
        读取所有
        :param path:
        :return:
        """
        if not os.path.exists(path):
            os.mkdir(path)
        for e in self._parameters:
            self._parameters[e] = torch.load(path + f'/{e}.torch_save')


class AttributeModel(nn.Module):
    def __init__(self, entity_count_a: int, entity_count_b: int, merged_rel_size: int,
                 encoded_charset: dict, device,
                 attr_lookup_table_a: torch.LongTensor, attr_lookup_table_b: torch.LongTensor, norm=1, dim=50,
                 max_char_length=50,
                 margin=1.0, use_lstm=True, train_first_set_a=True):
        # attr_lookup_table 存储每个属性编号对应的char编号，长度为max_char_length，填充0
        super(AttributeModel, self).__init__()
        self.device = device
        self.charset_size = len(encoded_charset)
        self.encoded_charset = encoded_charset
        self.norm = norm
        self.max_char_length = max_char_length
        self.features_dim = dim
        self.margin = margin
        self.use_lstm = use_lstm
        self.train_first_set_a = train_first_set_a
        self.merged_rel_size = merged_rel_size

        self.entity_count_a = entity_count_a
        self.entity_count_b = entity_count_b
        self.attr_lookup_table_a = attr_lookup_table_a[:, :max_char_length]
        self.attr_lookup_table_b = attr_lookup_table_b[:, :max_char_length]

        self.entities_emb_a = nn.Parameter(torch.Tensor(entity_count_a, dim))
        self.entities_emb_b = nn.Parameter(torch.Tensor(entity_count_b, dim))
        self.rel_embeddings = nn.Parameter(torch.Tensor(merged_rel_size, dim))
        self.char_embeddings = nn.Parameter(torch.Tensor(self.charset_size, dim))
        if use_lstm:
            self.lstm = nn.LSTM(dim, dim)

        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.reset_parameters()

    def attr_embedding(self, attr: torch.Tensor):
        """
        使用最长长度，避免损失信息，但是很大
        :param attr:B x self.max_char_length，存储字符串字符索引列表
        :return: attr嵌入
        """
        idx = attr.view(-1, 1)
        emb = self.char_embeddings[idx, :].view(self.max_char_length, attr.shape[0], self.features_dim)
        if self.use_lstm:
            output = self.lstm(emb)[0]
            output = output[-1, :, :]  # 获得最后一个输出
            return output
        else:
            output = torch.sum(emb, dim=0)
            return output

    def predict(self, triplets: torch.LongTensor, first_entities=True):
        """
        :param first_entities:
        :param triplets:Bx3代表h r t
        :return:
        """
        if first_entities:
            return self._distance(triplets, self.entities_emb_a, self.attr_lookup_table_a)
        else:
            return self._distance(triplets, self.entities_emb_b, self.attr_lookup_table_b)

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        if self.train_first_set_a:
            with torch.no_grad():
                self.entities_emb_a.data = self.entities_emb_a.data / torch.norm(self.entities_emb_a, p=2, dim=1).view(
                    -1, 1)
            pd = self._distance(positive_triplets, self.entities_emb_a, self.attr_lookup_table_a)
            nd = self._distance(negative_triplets, self.entities_emb_a, self.attr_lookup_table_a)
        else:
            with torch.no_grad():
                self.entities_emb_b.data = self.entities_emb_b.data / torch.norm(self.entities_emb_b, p=2, dim=1).view(
                    -1, 1)
            pd = self._distance(positive_triplets, self.entities_emb_b, self.attr_lookup_table_b)
            nd = self._distance(negative_triplets, self.entities_emb_b, self.attr_lookup_table_b)

        rt = torch.max(pd - nd + self.margin, torch.zeros(1, device=self.device))
        # rt = torch.max(pd + self.margin, torch.zeros(1, device=self.device))
        return rt
        # 不是一个值

    def reset_parameters(self):
        uniform_range = 6 / np.sqrt(self.features_dim)
        init.uniform_(self.rel_embeddings, -uniform_range, uniform_range)
        init.uniform_(self.entities_emb_a, -uniform_range, uniform_range)
        init.uniform_(self.entities_emb_b, -uniform_range, uniform_range)
        self.rel_embeddings.data = self.rel_embeddings / torch.norm(self.rel_embeddings, p=1, dim=1).view(-1, 1)
        # 初始化字符嵌入，使用onehot
        self.char_embeddings.data = torch.eye(self.charset_size, self.features_dim)

    def _distance(self, triplets: torch.LongTensor, entities_emb, attr_lookup_table):
        """
        distance 会在train的情况下，自动进行归一化实体嵌入
        :param triplets:
        :return:
        """
        assert triplets.size(1) == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        tails = attr_lookup_table[tails, :]
        return (entities_emb[heads, :] + self.rel_embeddings[relations, :]
                - self.attr_embedding(tails)).norm(p=self.norm, dim=1)

    def save(self, path):
        """
        存储所有
        :param path:
        :return:
        """
        # Module对象显式定义的Parameter类型的属性会放到self._parameters字典中,是有序字典
        if not os.path.exists(path):
            os.mkdir(path)
        for e in self._parameters:
            torch.save(self._parameters[e], path + f'/{e}.torch_save')

    def load(self, path):
        """
        读取所有
        :param path:
        :return:
        """
        if not os.path.exists(path):
            os.mkdir(path)
        for e in self._parameters:
            self._parameters[e] = torch.load(path + f'/{e}.torch_save')


class AttrModel(nn.Module):
    def __init__(self, gamma=1, relation_num=22, features_dim=50, use_lstm=False,
                 dis=l1_norm, device=None, max_char_length=100):
        super(AttrModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.charset_size = len(encoded_charset)
        self.char_embeddings = nn.Parameter(torch.Tensor(self.charset_size, features_dim))
        self.rel_attr_embeddings = nn.Parameter(torch.Tensor(relation_num, features_dim))
        self.dis = dis
        self.max_char_length = max_char_length
        self.input_samples_dim = relation_num
        self.features_dim = features_dim
        self.gamma = gamma

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.char_embeddings)
        nn.init.uniform_(self.rel_attr_embeddings)

    def forward(self, rel_attr_triples: list, entity_embeddings):
        """
        :param rel_attr_triples: 两个KG一起的属性元组
        """
        rt = []
        rel = []
        # 使用sum求和
        for triple in rel_attr_triples:
            rel.append([triple[0], triple[1]])
            aa = np.array([encoded_charset[i] for i in triple[-1]], dtype=np.int64)
            attr_embeddings = torch.index_select(self.char_embeddings, 0, torch.from_numpy(aa).to(self.device))
            attr_embeddings = torch.sum(attr_embeddings, dim=0)
            rt.append(attr_embeddings.view(1, -1))
        rel = torch.from_numpy(np.array(rel, dtype=np.int64)).to(self.device)
        t = torch.cat(rt, dim=0)
        h = torch.index_select(rel, 1, torch.tensor([0]).to(self.device))  # 获得第1列
        h = torch.index_select(entity_embeddings, 0, h.view([h.shape[0]]).long())  # view变成标量
        r = torch.index_select(rel, 1, torch.tensor([1]).to(self.device))  # 获得第2列
        r = torch.index_select(self.rel_attr_embeddings, 0, r.view([t.shape[0]]).type(torch.int64))  # view变成标量
        rt = torch.max(self.dis(h + r - t) + self.gamma, torch.Tensor([0]).to(self.device))
        return torch.sum(rt)

    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        for e in self._parameters:
            torch.save(self._parameters[e], path + f'/{e}.torch_save')

    def load(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        for e in self._parameters:
            self._parameters[e] = torch.load(path + f'/{e}.torch_save')


# 单独的loss
class JointLearning1(nn.Module):
    def __init__(self, char_embeddings, entity_embeddings, lstm=None):
        super(JointLearning1, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.char_embeddings = char_embeddings
        # self.entity_emb_KG1 = entity_emb_KG1
        # self.entity_emb_KG2 = entity_emb_KG2
        self.entity_embeddings = entity_embeddings
        self.lstm = lstm

    def forward(self, rel_attr_triples: list):
        """
        对于所有属性三元组，就散cos 相似度，使用attr嵌入
        :return:
        """
        if self.lstm:
            # 使用lstm
            pass
        else:
            rt = []
            rel = []
            # 使用sum求和
            for triple in rel_attr_triples:
                rel.append([triple[0], triple[1]])
                aa = np.array([encoded_charset[i] for i in triple[-1]], dtype=np.int64)
                attr_embeddings = torch.index_select(self.char_embeddings, 0, torch.from_numpy(aa).to(self.device))
                attr_embeddings = torch.sum(attr_embeddings, dim=0)
                rt.append(attr_embeddings.view(1, -1))
            rel = torch.from_numpy(np.array(rel, dtype=np.int64)).to(self.device)
            t = torch.cat(rt, dim=0)
            h = torch.index_select(rel, 1, torch.tensor([0]).to(self.device))  # 获得第1列
            h = torch.index_select(self.entity_embeddings, 0, h.view([h.shape[0]]).long())  # view变成标量
            # h t
            h = h / torch.norm(h)
            t = t / torch.norm(t)
            # rt = 1 - cos
            return torch.sum(1 - torch.sum(h * t, dim=1))


class JointLearning2(nn.Module):
    def __init__(self, gamma=1, relation_num=22, features_dim=50, use_lstm=False,
                 dis=l1_norm, device=None, max_char_length=100,
                 input_samples_dim1=None, input_samples_dim2=None):
        super(JointLearning2, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.charset_size = len(encoded_charset)
        self.KG1 = Model(entity_num=input_samples_dim1, margin=gamma, features_dim=features_dim, dis=dis,
                         device=device).to(device)
        self.KG2 = Model(entity_num=input_samples_dim2, margin=gamma, features_dim=features_dim, dis=dis,
                         device=device).to(device)
        self.attr = AttrModel(gamma=gamma, use_lstm=use_lstm, relation_num=relation_num, dis=dis,
                              features_dim=features_dim,
                              max_char_length=max_char_length, device=device).to(device)

        self.dis = dis
        self.max_char_length = max_char_length
        self.input_samples_dim = relation_num
        self.features_dim = features_dim
        self.gamma = gamma

    def forward(self, rel_triples1: torch.Tensor, rel_triples2: torch.Tensor,
                rel_attr_triples1: list, rel_attr_triples2: list):
        ls1 = self.KG1(rel_triples1)
        print(ls1)
        ls2 = self.KG2(rel_triples2)
        print(ls2)
        ls3 = self.attr(rel_attr_triples1, self.KG1.entity_embeddings) + self.attr(rel_attr_triples2,
                                                                                   self.KG2.entity_embeddings)
        print(ls3)
        # 合并损失
        rt = []
        rel = []
        # 使用sum求和
        for triple in rel_attr_triples1:
            rel.append([triple[0], triple[1]])
            aa = np.array([encoded_charset[i] for i in triple[-1]], dtype=np.int64)
            attr_embeddings = torch.index_select(self.attr.char_embeddings, 0, torch.from_numpy(aa).to(self.device))
            attr_embeddings = torch.sum(attr_embeddings, dim=0)
            rt.append(attr_embeddings.view(1, -1))
        rel = torch.from_numpy(np.array(rel, dtype=np.int64)).to(self.device)
        t = torch.cat(rt, dim=0)
        h = torch.index_select(rel, 1, torch.tensor([0]).to(self.device))  # 获得第1列
        h = torch.index_select(self.KG1.entity_embeddings, 0, h.view([h.shape[0]]).long())  # view变成标量
        # h t
        h = h / torch.norm(h)
        t = t / torch.norm(t)
        # rt = 1 - cos
        ls4 = torch.sum(1 - torch.sum(h * t, dim=1))

        rt = []
        rel = []
        for triple in rel_attr_triples2:
            rel.append([triple[0], triple[1]])
            aa = np.array([encoded_charset[i] for i in triple[-1]], dtype=np.int64)
            attr_embeddings = torch.index_select(self.attr.char_embeddings, 0, torch.from_numpy(aa).to(self.device))
            attr_embeddings = torch.sum(attr_embeddings, dim=0)
            rt.append(attr_embeddings.view(1, -1))
        rel = torch.from_numpy(np.array(rel, dtype=np.int64)).to(self.device)
        t = torch.cat(rt, dim=0)
        h = torch.index_select(rel, 1, torch.tensor([0]).to(self.device))  # 获得第1列
        h = torch.index_select(self.KG2.entity_embeddings, 0, h.view([h.shape[0]]).long())  # view变成标量
        h = h / torch.norm(h)
        t = t / torch.norm(t)
        ls4 += torch.sum(1 - torch.sum(h * t, dim=1))
        return ls1 + ls2 + ls3 + ls4

    def save(self, path):
        self.KG1.save(path)
        self.KG2.save(path)
        self.attr.save(path)

    def load(self, path):
        self.KG1.load(path)
        self.KG2.load(path)
        self.attr.load(path)


# 合并的总的loss

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_generator = RelationDataset(attr_rel_tw)
    train_generator = torch.utils.data.DataLoader(train_generator, batch_size=64)
    model1 = Model(entity_count=len(id2URL_tw), relation_count=len(id2URL_tw), device=device, alpha=alpha_tw).to(device)
    aaa = torch.Tensor(attrs_only_tw).long().to(device)
    model2 = AttributeModel(len(attr_rel_tw), encoded_charset, device, aaa).to(device)
    # optimizer = torch.optim.Adam([model1.entities_emb, model2.rel_embeddings, model2.char_embeddings], lr=1e-3)
    # model2.register_parameter('eee1', model1.entities_emb)
    optimizer = torch.optim.Adam(list(model2.parameters()) + [model1.entities_emb], lr=1e-3)
    for i in train_generator:
        # print(model.attr_embedding(aaa[i[-1], :]))
        optimizer.zero_grad()
        triples = torch.stack([i[0], i[1], i[2]], dim=1)
        loss = model2(triples, triples, model1.entities_emb)
        print(loss.sum().item())
        loss.mean().backward()
        optimizer.step()
    # rel_triples = torch.tensor(rel_tw, dtype=torch.int32, device=device)
    # mxx = max([int(i) for i in id2URL_tw])
    # model = Model(input_samples_dim=mxx + 1).to(device)  # 因为编号不是从0开始，所以要有空闲部分
    # model1 = AttrModel(entity_embeddings=model.entity_embeddings).to(device)
    # print(model1(attr_rel_tw))
    # j = JointLearning2(model1.char_embeddings, model.entity_embeddings)
    # print(j(attr_rel_tw))

    # mxx1 = max([int(i) for i in id2URL_tw]) + 1
    # mxx2 = max([int(i) for i in id2URL_fq]) + 1
    # mm = JointLearning2(input_samples_dim1=mxx1, input_samples_dim2=mxx2, device=device)
    # print(mm(rel_tw, rel_fq, attr_rel_tw, attr_rel_fq))
    # for i in attr_rel_tw:
    #     if int(i[0]) > model1.entity_embeddings.shape[0]:
    #         print(i)
    # attr_rel_tw = torch.tensor(attr_rel_tw, dtype=torch.int32, device=device)
    # 应该合并的。。
    # p=0
    # for i in dl:
    #     p=i
    #     print(i)
    #     print(type(i))
    #     break
    # batch = 100
    # sep = len(attr_rel_tw) // batch
    # model1 = AttrModel(entity_embeddings=model.entity_embeddings).to(device)
    # for i in range(batch):
    #     data = attr_rel_tw[i * sep:(i + 1) * sep]
    #     print(torch.mean(model1(data)).item())
    # model.to(device)会自动把所有的self。xxxtensor变成.to(device)
