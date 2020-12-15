import os

import torch.nn as nn
import torch.nn.init as init

from distance import *
from load_data import encoded_charset
from uitls import *

best_gamma = [1, 5, 10]
best_dim = [50, 75, 100, 200]
best_learning_rate = [0.001, 0.01, 0.1]


# TODO
# self.attr_embedding=nn.Parameter(torch.Tensor(, features_dim))
# self.char_embedding = nn.Parameter(torch.Tensor(input_samples, features_dim))

class Model(nn.Module):
    def __init__(self, gamma=1, input_samples_dim=None, features_dim=50,
                 dis=l1_norm, device=None):
        """
        :param input_samples_dim: 输入维度
        :param gamma: 超参数
        :param features_dim:特征维度，矩阵的列数
        :param use_f_type: 属性嵌入的聚合函数
        """
        super(Model, self).__init__()
        assert input_samples_dim is not None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

        # 指定大小，如果写[xx,xx]就是指定值了
        self.entity_embeddings = nn.Parameter(torch.Tensor(input_samples_dim, features_dim))
        self.rel_embeddings = nn.Parameter(torch.Tensor(1, features_dim))
        self.dis = dis
        self.input_samples_dim = input_samples_dim
        self.features_dim = features_dim
        self.gamma = gamma

        self.reset_parameters()

    def forward(self, rel_triples: torch.Tensor, cor_rel_triples):
        if not isinstance(rel_triples, torch.Tensor):
            rel_triples = torch.Tensor(rel_triples).to(self.device).type(torch.int32)
        # l2正则化
        # self.entity_embeddings.data = self.entity_embeddings / torch.norm(self.entity_embeddings, dim=1). \
        #     view(self.entity_embeddings.shape[0], 1).to(self.device)

        r = self.rel_embeddings  # 只有一个r
        h = rel_triples[:, 0].long()
        h = self.entity_embeddings[h, :]
        t = rel_triples[:, 1].long()
        t = self.entity_embeddings[t, :]
        # 会自动广播，导致维度一致
        ch, cr, ct = get_structure_corrupted(h, r, t, self.entity_embeddings, self.device)
        # zero = torch.zeros(self.input_samples_dim, 1).to(self.device)
        rt = torch.max(self.dis(h + r - t) - self.dis(ch + cr - ct) + self.gamma, torch.zeros(1).to(self.device))
        # rt为 a x 1,表示每一组h r t关系的距离
        # rt = torch.max(self.dis(h + r - t) - self.dis(ch + cr - ct) + self.gamma, torch.Tensor([0]).to(self.device))
        return torch.sum(rt, dim=0)

    def reset_parameters(self):
        # 更适用与relu,默认使用kaiming
        # init.kaiming_uniform()
        # tanh使用Xavier
        # nn.init.kaiming_uniform(self.entity_embeddings)
        uniform_range = 6 / np.sqrt(self.features_dim)
        init.uniform_(self.entity_embeddings, -uniform_range, uniform_range)
        init.uniform_(self.rel_embeddings, -uniform_range, uniform_range)
        nn.Embedding
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


# max_char_length 表示属性对多这么长，后面会忽略


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
        self.KG1 = Model(gamma=gamma, dis=dis, features_dim=features_dim, input_samples_dim=input_samples_dim1,
                         device=device).to(device)
        self.KG2 = Model(gamma=gamma, dis=dis, features_dim=features_dim, input_samples_dim=input_samples_dim2,
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
