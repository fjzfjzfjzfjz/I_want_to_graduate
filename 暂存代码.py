# class AttrModel(nn.Module):
#     def __init__(self, gamma=1, relation_num=22, features_dim=50, use_f_type=None,
#                  dis=l1_norm, device=None, max_char_length=25, entity_embeddings=None):
#         super(AttrModel, self).__init__()
#         assert use_f_type in [None, 'sum', 'lstm']
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
#         self.charset_size = len(encoded_charset)
#         self.char_embeddings = nn.Parameter(torch.Tensor(self.charset_size, features_dim))
#         self.rel_attr_embeddings = nn.Parameter(torch.Tensor(relation_num, features_dim))
#         self.lstm = nn.LSTM(features_dim, features_dim)  # 注意lstm的输入格式
#         # 输入特征数和输出特征数，lstm cell的个数由输入序列的长度决定
#         self.dis = dis
#         self.max_char_length = max_char_length
#         self.input_samples_dim = relation_num
#         self.features_dim = features_dim
#         self.gamma = gamma
#         self.entity_embeddings = entity_embeddings
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.uniform_(self.char_embeddings)
#         nn.init.uniform_(self.rel_attr_embeddings)
#
#     def forward(self, rel_attr_triples: list):
#         """
#         :param rel_attr_triples: 两个KG一起的属性元组
#         """
#         # 获得属性嵌入
#         rt = []
#         rel = []
#         for triple in rel_attr_triples:
#             # 提取字符嵌入
#             # print(triple)
#             rel.append([triple[0], triple[1]])
#             aa = np.array([encoded_charset[i] for i in triple[-1]], dtype=np.int64)
#             if len(aa) < self.max_char_length:
#                 aa = np.concatenate([aa, np.array([0] * (self.max_char_length - len(aa)))], axis=0)
#             else:
#                 aa = aa[:self.max_char_length]
#             attr_embeddings = torch.index_select(self.char_embeddings, 0, torch.from_numpy(aa).to(self.device))
#             rt.append(attr_embeddings)
#             # output = self.lstm(attr_embeddings.view(len(triple[-1]), 1, 50))
#             # rt.append(output[0][-1, :, :])
#         rel = torch.from_numpy(np.array(rel, dtype=np.int64)).to(self.device)
#         inputs = torch.cat(rt, dim=0)
#         print(inputs.shape)
#         t = self.lstm(inputs.view(self.max_char_length, len(rel_attr_triples), 50))[0][-1, :, :]
#         # t = torch.cat(rt, dim=0)
#         r = torch.index_select(rel, 1, torch.tensor([1]).to(self.device))  # 获得第2列
#         r = torch.index_select(self.rel_attr_embeddings, 0, r.view([t.shape[0]]).type(torch.int64))  # view变成标量
#         h = torch.index_select(rel, 1, torch.tensor([0]).to(self.device))  # 获得第1列
#         h = torch.index_select(self.entity_embeddings, 0, h.view([h.shape[0]]).long())  # view变成标量
#         # 会自动广播，导致维度一致
#         # ch, cr, ct = get_structure_corrupted(h, r, t, self.entity_embeddings, self.device, rel_triples)
#         # zero = torch.zeros(self.input_samples_dim, 1).to(self.device)
#         # rt = torch.max(self.dis(h + r - t) - self.dis(ch + cr - ct) + self.gamma, torch.zeros(1).to(self.device))
#         rt = torch.max(self.dis(h + r - t) + self.gamma, torch.Tensor([0]).to(self.device))
#         return torch.sum(rt, dim=0)
#
#     def save(self, path):
#         if not os.path.exists(path):
#             os.mkdir(path)
#         for e in self._parameters:
#             torch.save(self._parameters[e], path + f'/{e}.torch_save')
#
#     def load(self, path):
#         if not os.path.exists(path):
#             os.mkdir(path)
#         for e in self._parameters:
#             self._parameters[e] = torch.load(path + f'/{e}.torch_save')




# class AttrModel(nn.Module):
#     def __init__(self, gamma=1, relation_num=22, features_dim=50, use_f_type=None,
#                  dis=l1_norm, device=None, max_char_length=100, entity_embeddings=None):
#         super(AttrModel, self).__init__()
#         assert use_f_type in [None, 'sum', 'lstm']
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
#         self.charset_size = len(encoded_charset)
#         self.char_embeddings = nn.Parameter(torch.Tensor(self.charset_size, features_dim))
#         self.rel_attr_embeddings = nn.Parameter(torch.Tensor(relation_num, features_dim))
#         self.lstm = nn.LSTM(features_dim, features_dim)  # 注意lstm的输入格式
#         # 输入特征数和输出特征数，lstm cell的个数由输入序列的长度决定
#         self.dis = dis
#         self.max_char_length = max_char_length
#         self.input_samples_dim = relation_num
#         self.features_dim = features_dim
#         self.gamma = gamma
#         self.entity_embeddings = entity_embeddings
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.uniform_(self.char_embeddings)
#         nn.init.uniform_(self.rel_attr_embeddings)
#
#     def forward(self, rel_attr_triples: list):
#         """
#         :param rel_attr_triples: 两个KG一起的属性元组
#         """
#         rt = []
#         rel = []
#         for triple in rel_attr_triples:
#             rel.append([triple[0], triple[1]])
#             aa = np.array([encoded_charset[i] for i in triple[-1]], dtype=np.int64)
#             attr_embeddings = torch.index_select(self.char_embeddings, 0, torch.from_numpy(aa).to(self.device))
#             output = self.lstm(attr_embeddings.view(len(triple[-1]), 1, 50))
#             pp = output[0][-1, :, :]
#             del output
#             rt.append(pp)
#         rel = torch.from_numpy(np.array(rel, dtype=np.int64)).to(self.device)
#         t = torch.cat(rt, dim=0)
#         h = torch.index_select(rel, 1, torch.tensor([0]).to(self.device))  # 获得第1列
#         h = torch.index_select(self.entity_embeddings, 0, h.view([h.shape[0]]).long())  # view变成标量
#         r = torch.index_select(rel, 1, torch.tensor([1]).to(self.device))  # 获得第2列
#         r = torch.index_select(self.rel_attr_embeddings, 0, r.view([t.shape[0]]).type(torch.int64))  # view变成标量
#         rt = torch.max(self.dis(h + r - t) + self.gamma, torch.Tensor([0]).to(self.device))
#         return rt
#
#     def save(self, path):
#         if not os.path.exists(path):
#             os.mkdir(path)
#         for e in self._parameters:
#             torch.save(self._parameters[e], path + f'/{e}.torch_save')
#
#     def load(self, path):
#         if not os.path.exists(path):
#             os.mkdir(path)
#         for e in self._parameters:
#             self._parameters[e] = torch.load(path + f'/{e}.torch_save')



# def get_structure_corrupted(h, r, t, entity_embedding: torch.Tensor, device, rel_dict: dict):
#     """
#     使用api 加速
#     :param entity_embedding: 某个KG的关注关系集合,[a,b]
#     :param h:
#     :param r:只有一个关系，一行
#     :param t:
#     :return: 替换成这个关系不具备的其他的;随机替换头尾
#     """
#     head = np.random.random() < 0.5
#     # print('head ', head)
#     # TODO 去除错误的反例，即保证反例不会恰好是正例
#     if head:
#         idxs = np.random.randint(0, entity_embedding.shape[0], (h.shape[0],), dtype=np.int64)
#         ch = torch.index_select(entity_embedding, 0, torch.from_numpy(idxs).to(device))
#         return ch, r, t
#     else:
#         idxs = np.random.randint(0, entity_embedding.shape[0], (h.shape[0],), dtype=np.int64)
#         ct = torch.index_select(entity_embedding, 0, torch.from_numpy(idxs).to(device))
#         return h, r, ct
#     # 产生随机数本身不慢，可以用for
#     # if head:
#     #     cc = []
#     #     for i in range(h.shape[0]):
#     #         tt = np.random.randint(0, entity_embedding.shape[0], dtype=np.int64)
#     #         tt_embedding=torch.index_select(entity_embedding, 0, torch.from_numpy(tt).to(device))
#     #         while True:
#     #             if t not in rel_dict or (t in rel_dict ):break
#     #             if tt_embedding
#     #     idxs = np.random.randint(0, entity_embedding.shape[0], (h.shape[0],), dtype=np.int64)
#     #     ch = torch.index_select(entity_embedding, 0, torch.from_numpy(idxs).to(device))
#     #     return ch, r, t
#     # else:
#     #     idxs = np.random.randint(0, entity_embedding.shape[0], (h.shape[0],), dtype=np.int64)
#     #     ct = torch.index_select(entity_embedding, 0, torch.from_numpy(idxs).to(device))
#     #     return h, r, ct
#
#     # for i in range(d.shape[0]):
#     #     if i % 10000 == 0: print(i)
#     #     cc = np.random.choice(idxs)
#     # while cc == i:
#     #     cc = np.random.choice(idxs)
#     # TODO 因随机到重复的概率很低，所以不考虑重复
#     # TODO 反例不应h+r=t，应该排除，否则会难以收敛
#     # d[i] = entity_embedding[cc]
#     # return h, r, torch.Tensor([1, 2, 3])  # 作为桩模块
#     # if head:
#     #     return d, r, t
#     # else:
#     #     return h, r, d



# def get_structure_corrupted(h, r, t, entity_embedding: torch.Tensor, device, cor_rel: dict):
#     """
#     使用api 加速
#     :param device:
#     :param cor_rel:
#     :param entity_embedding: 某个KG的关注关系集合,[a,b]
#     :param h:
#     :param r:只有一个关系，一行
#     :param t:
#     :return: 替换成这个关系不具备的其他的;随机替换头尾
#     """
#     head = cor_rel['head']
#     rel = np.array(cor_rel['col'], dtype=np.int64)
#     # print('head ', head)
#     if head:
#         ch = torch.index_select(entity_embedding, 0, torch.from_numpy(rel).to(device))
#         return ch, r, t
#     else:
#         ct = torch.index_select(entity_embedding, 0, torch.from_numpy(rel).to(device))
#         return h, r, ct


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
    for triple in tt:
        h = entity_embeddings[int(triple[0])]
        score = dis(h + r - entity_embeddings)
        _, indices = torch.sort(score)
        # 从indices找triple[1]是第几个
        _, indices = torch.sort(indices)
        # print(indices[triple[1]])
        r = indices[triple[1]].item() + 1
        # 得+1，因为<=10
        if r <= 10:
            hits10 += 1
        rank += r
        ii += 1
        if ii % 1000 == 0:
            print(ii, ll, hits10 / ii)
    return hits10 / sample_size, rank / sample_size
