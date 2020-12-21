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


# 预处理出cor，即每一个rel 三元组的cor，否则速度太慢了
# 使用总体规定头尾，进行加速
def prepare_cor(rel: list, id2URL: dict, is_attr=False):
    head = np.random.random() < 0.5
    rt = []
    idxs = [int(i) for i in id2URL]
    ii = 0
    total = 0
    # print('head ', head)
    # TODO 去除错误的反例，即保证反例不会恰好是正例
    # 首先制作点的出边集合，便于判断
    if is_attr:
        dd = {}
        for i in rel:
            if i[0] not in dd:
                dd[i[0]] = {i[1]: i[2]}
            else:
                dd[i[0]][i[1]] = i[2]
        attrs = [i[-1] for i in rel]
        for i in rel:
            ii += 1
            if ii % 10000 == 0:
                print(f'{ii * 100 / len(rel)}% {ii + 1},{len(rel)}')
            if head:
                cc = np.random.choice(idxs, (1,))[0]
                while cc in dd and i[1] in dd[cc] and dd[cc][i[1]] == i[2]:
                    cc = np.random.choice(idxs, (1,))[0]
                rt.append([cc, i[1], i[2]])
            else:
                # 属性几乎不可能选到重复的
                cc = np.random.choice(attrs)
                rt.append([i[0], i[1], cc])
    else:
        dd = {}
        for i in rel:
            if i[0] not in dd:
                dd[i[0]] = set()
                dd[i[0]].add(i[1])
            else:
                dd[i[0]].add(i[1])
        for i in rel:
            ii += 1
            if ii % 10000 == 0:
                print(f'{ii * 100 / len(rel)}% {ii},{len(rel)}')
            if head:
                cc = np.random.choice(idxs)
                while cc in dd and i[1] in dd[cc]:
                    cc = np.random.choice(idxs)
                    total += 1
                rt.append([cc, i[1]])
            else:
                cc = np.random.choice(idxs)
                while i[0] in dd and cc in dd[i[0]]:
                    cc = np.random.choice(idxs)
                    total += 1
                rt.append([i[0], cc])
    print(f'重复 {total} 次')
    return rt, head


def prepare_cor_wrapper(rel: list, id2URL: dict, is_attr=False):
    # 调用prepare_cor，并制成，{head：bool,col:list}，只获得替换的那一列，按顺序
    rt, head = prepare_cor(rel, id2URL, is_attr)
    # print([i[-1] for i in rt])
    if head:
        return {'head': head,
                'col': [i[0] for i in rt]}
    else:
        return {'head': head,
                'col': [i[-1] for i in rt]}


# rel_fq_cor = prepare_cor_wrapper(rel_fq, id2URL_fq, False)
# rel_tw_cor = prepare_cor_wrapper(rel_tw, id2URL_tw, False)
# rel_fb_cor = prepare_cor(rel_fb, id2URL_fb, False)

# pickle.dump(rel_fq_cor, open('preprocessed_data/foursquare/rel_fq_cor.pickle', 'wb'))
# pickle.dump(rel_tw_cor, open('preprocessed_data/twitter/rel_tw_cor.pickle', 'wb'))
# pickle.dump(rel_fb_cor, open('preprocessed_data/facebook/rel_fb_cor.pickle', 'wb'))
# attr_rel_fq_cor = prepare_cor(attr_rel_fq, id2URL_fq, True)
# attr_rel_tw_cor = prepare_cor(attr_rel_tw, id2URL_tw, True)
# attr_rel_fb_cor = prepare_cor(attr_rel_fb, id2URL_fb, True)

# rel_tw_cor = pickle.load(open('preprocessed_data/twitter/rel_tw_cor.pickle', 'rb'))
# rel_fq_cor = pickle.load(open('preprocessed_data/foursquare/rel_fq_cor.pickle', 'rb'))
# rel_fb_cor = pickle.load(open('preprocessed_data/facebook/rel_fb_cor.pickle', 'rb'))


def train(epoch=1000, lr=1e-3, dis=l1_norm, margin=1, dim=50):
    rel_tw1 = np.array(rel_tw, dtype=np.int64)
    rel_tw1 = np.stack([rel_tw1[:, 0], np.zeros([rel_tw1.shape[0]], dtype=np.int64), rel_tw1[:, 1]], axis=1)
    rel_triples = torch.tensor(rel_tw1, dtype=torch.int64, device=device)
    mxx = max([int(i) for i in id2URL_tw]) + 1
    rel_tw1_cor = np.stack([rel_tw1[:, 0], np.zeros([rel_tw1.shape[0]], dtype=np.int64), np.array(rel_tw_cor['col'])],
                           axis=1)
    rel_cor = torch.tensor(rel_tw1_cor, dtype=torch.int64, device=device)
    model = TransE(input_samples_dim=mxx, dis=dis, features_dim=dim, margin=margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ls = []
    for i in range(epoch):
        optimizer.zero_grad()
        loss = model(rel_triples, rel_cor)[0].sum()
        # print(model.rel_embeddings)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('epoch %d ,loss %.2f' % (i + 1, loss.item()))
        ls.append(loss.item())
        if (i + 1) % 100 == 0:
            hits10, mr = structure_hits10_and_mean_rank(rel_triples, model.entity_embeddings, model.rel_embeddings,
                                                        device)
            print(f'{hits10 * 100}% {mr}')
    plot_loss(ls, range(1, epoch + 1))
    model.save('model_checkpoint/test/')


def train_tw_entity(epoch=1000, lr=1e-3, dis=l1_norm, margin=1, dim=75, norm=1):
    # rel_triples = torch.tensor(rel, dtype=torch.int32, device=device)
    # mxx = max([int(i) for i in id2URL_tw]) + 1
    mxx = len(id2URL_tw)
    mxx1 = len(id2URL_tw)
    model = Model(entity_count=mxx, relation_count=mxx1, margin=margin, dim=dim, norm=norm, device=device).to(
        device)  # 因为编号不是从0开始，所以要有空闲部分

    # model.to(device)会自动把所有的self。xxxtensor变成.to(device)
    # model.load('model_checkpoint/test/')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ls = []
    for i in range(epoch):
        optimizer.zero_grad()
        pos_triples = torch.Tensor(rel_tw).to(device).long()
        neg_triples = get_structure_corrupted(pos_triples, model.entities_emb, device)
        loss = model(pos_triples, neg_triples)
        # print(model.rel_embeddings)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print('epoch %d ,loss %.2f' % (i + 1, loss.item()))
        ls.append(loss.item())
        if (i + 1) % 50 == 0:
            test1 = torch.Tensor(rel_tw).to(device).long()
            hits10, mr = structure_hits10_and_mean_rank(model, test1, model.entity_count, device)

            print(f'{hits10 * 100}% {mr}')
    plot_loss(ls, range(1, epoch + 1))
    model.save('model_checkpoint/test/')


def train_attr(epoch=10, lr=1e-3, dis=l1_norm):
    mxx = max([int(i) for i in id2URL_tw]) + 1
    model = Model(entity_num=mxx, dis=dis).to(device)
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
    model = Model(entity_num=mxx, dis=dis).to(device)
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

# class Model(nn.Module):
#     def __init__(self, entity_num, rel_num, margin=1, features_dim=50,
#                  dis=l1_norm, device=None):
#         super(Model, self).__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
#
#         # 指定大小，如果写[xx,xx]就是指定值了
#         self.entity_embeddings = nn.Parameter(torch.Tensor(entity_num, features_dim))
#         self.rel_embeddings = nn.Parameter(torch.Tensor(rel_num, features_dim))
#         self.dis = dis
#         self.entity_num = entity_num
#         self.rel_num = rel_num
#         self.features_dim = features_dim
#         self.margin = margin
#         self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
#
#         self.reset_parameters()
#
#     def forward(self, rel_triples: torch.Tensor):
#         # l2正则化
#         self.entity_embeddings.data = self.entity_embeddings / torch.norm(self.entity_embeddings, p=2, dim=1,
#                                                                           keepdim=True)
#
#         r = rel_triples[:, 1].long()
#         r = self.rel_embeddings[r, :]
#         h = rel_triples[:, 0].long()
#         h = self.entity_embeddings[h, :]
#         t = rel_triples[:, 2].long()
#         t = self.entity_embeddings[t, :]
#         ch, cr, ct = get_structure_corrupted(h, r, t, self.entity_embeddings, self.device)
#         # rt为 a x 1,表示每一组h r t关系的距离
#         rt = torch.max(self.dis(h + r - t) - self.dis(ch + cr - ct) + self.margin, torch.Tensor([0]).to(self.device))
#         return torch.sum(rt, dim=0)
#         # return rt
#
#     def reset_parameters(self):
#         uniform_range = 6 / np.sqrt(self.features_dim)
#         init.uniform_(self.entity_embeddings, -uniform_range, uniform_range)
#         # self.entity_embeddings.data = self.entity_embeddings / torch.norm(self.entity_embeddings, dim=1).view(-1, 1)
#         init.uniform_(self.rel_embeddings, -uniform_range, uniform_range)
#         self.rel_embeddings.data = self.rel_embeddings / torch.norm(self.rel_embeddings, p=1, dim=1).view(-1, 1)
#
#     def save(self, path):
#         """
#         存储所有
#         :param path:
#         :return:
#         """
#         # Module对象显式定义的Parameter类型的属性会放到self._parameters字典中,是有序字典
#         if not os.path.exists(path):
#             os.mkdir(path)
#         for e in self._parameters:
#             torch.save(self._parameters[e], path + f'/{e}.torch_save')
#
#     def load(self, path):
#         """
#         读取所有
#         :param path:
#         :return:
#         """
#         if not os.path.exists(path):
#             os.mkdir(path)
#         for e in self._parameters:
#             self._parameters[e] = torch.load(path + f'/{e}.torch_save')


# max_char_length 表示属性对多这么长，后面会忽略
# class Model(nn.Module):
#     def __init__(self, entity_count, relation_count, device, norm=1, dim=100, margin=1.0, dis=l1_norm):
#         super(Model, self).__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
#
#         # 指定大小，如果写[xx,xx]就是指定值了
#         self.entities_emb = nn.Parameter(torch.Tensor(entity_count, dim))
#         self.rel_embeddings = nn.Parameter(torch.Tensor(relation_count, dim))
#         self.dis = dis
#         self.input_samples_dim = entity_count
#         self.features_dim = dim
#         self.gamma = margin
#         self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
#
#         self.reset_parameters()
#
#     def predict(self, triplets: torch.LongTensor):
#         r = triplets[:, 1]
#         r = self.rel_embeddings[r, :]
#         h = triplets[:, 0]
#         h = self.entities_emb[h, :]
#         t = triplets[:, 2]
#         t = self.entities_emb[t, :]
#         return self.dis(h + r - t)
#
#     def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
#         self.entities_emb.data = self.entities_emb / torch.norm(self.entities_emb, dim=1, p=2).view(-1, 1)
#
#         r = positive_triplets[:, 1]
#         r = self.rel_embeddings[r, :]
#         h = positive_triplets[:, 0]
#         h = self.entities_emb[h, :]
#         t = positive_triplets[:, 2]
#         t = self.entities_emb[t, :]
#
#         cr = negative_triplets[:, 1]
#         cr = self.rel_embeddings[cr, :]
#         ch = negative_triplets[:, 0]
#         ch = self.entities_emb[ch, :]
#         ct = negative_triplets[:, 2]
#         ct = self.entities_emb[ct, :]
#         # ch, cr, ct = get_structure_corrupted(h, r, t, self.entities_emb, self.device)
#         pd = self.dis(h + r - t)
#         nd = self.dis(ch + cr - ct)
#         rt = torch.max(pd - nd + self.gamma, torch.zeros(1).to(self.device))
#         # target = torch.tensor([-1], dtype=torch.long, device=self.device)
#         # rt = self.criterion(self.dis(h + r - t), self.dis(ch + cr - ct), target)
#         return rt
#         # 不是一个值
#
#     def reset_parameters(self):
#         uniform_range = 6 / np.sqrt(self.features_dim)
#         init.uniform_(self.entities_emb, -uniform_range, uniform_range)
#         init.uniform_(self.rel_embeddings, -uniform_range, uniform_range)
#         self.rel_embeddings.data = self.rel_embeddings / torch.norm(self.rel_embeddings, p=1, dim=1).view(-1, 1)
#
#     def _distance(self):
#
#
#     def save(self, path):
#         """
#         存储所有
#         :param path:
#         :return:
#         """
#         # Module对象显式定义的Parameter类型的属性会放到self._parameters字典中,是有序字典
#         if not os.path.exists(path):
#             os.mkdir(path)
#         for e in self._parameters:
#             torch.save(self._parameters[e], path + f'/{e}.torch_save')
#
#     def load(self, path):
#         """
#         读取所有
#         :param path:
#         :return:
#         """
#         if not os.path.exists(path):
#             os.mkdir(path)
#         for e in self._parameters:
#             self._parameters[e] = torch.load(path + f'/{e}.torch_save')


class AttributeModel(nn.Module):
    def __init__(self, entity_count_a: int, entity_count_b: int, merged_rel_size: int,
                 encoded_charset: dict, device,
                 attr_lookup_table_a: torch.LongTensor, attr_lookup_table_b: torch.LongTensor, norm=1, dim=50,
                 max_char_length=160, offset=1000000,
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
        self.offset = offset  # 百万，b的索引会+一百万
        assert self.offset > entity_count_a and self.offset > entity_count_b

        self.entity_count_a = entity_count_a
        self.entity_count_b = entity_count_b
        self.attr_lookup_table_a = attr_lookup_table_a
        self.attr_lookup_table_b = attr_lookup_table_b

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

    def predict(self, triplets: torch.LongTensor):
        """
        :param triplets:Bx3代表h r t
        :return:
        """
        return self._distance(triplets)

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        pd = self._distance(positive_triplets, True)
        nd = self._distance(negative_triplets, True)
        rt = torch.max(pd - nd + self.margin, torch.zeros(1, device=self.device))
        # target = torch.tensor([-1], dtype=torch.long, device=self.device)
        # rt = self.criterion(self.dis(h + r - t), self.dis(ch + cr - ct), target)
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

    def _distance(self, triplets: torch.LongTensor, training=False):
        """
        distance 会在train的情况下，自动进行归一化实体嵌入
        :param triplets:
        :return:
        """
        # 根据实体编号、attr编号得知是a还是b
        assert triplets.size(1) == 3
        if training:
            self.normalize(triplets)

        # 获得类型,因为数据集按照a b顺序输出所以
        heads = triplets[:, 0]

        mask_a = torch.nonzero(heads < self.offset)[:, 0]  # 行号
        mask_b = torch.nonzero(heads >= self.offset)[:, 0]

        heads_a = triplets[mask_a, 0]
        heads_b = triplets[mask_b, 0]
        heads = torch.cat([self.entities_emb_a[heads_a, :], self.entities_emb_b[heads_b, :]], dim=0)

        tails_a = triplets[mask_a, 2]
        tails_a = self.attr_lookup_table_a[tails_a, :]
        tails_b = triplets[mask_b, 2]
        tails_b = self.attr_lookup_table_b[tails_b, :]
        tails = torch.cat([self.attr_embedding(tails_a), self.attr_embedding(tails_b)], dim=0)

        relations = triplets[:, 1]
        relations = self.rel_embeddings[relations, :]
        # tails = attr_lookup_table[tails, :]
        # return (entities_emb[heads, :] + self.rel_embeddings[relations, :]
        #         - self.attr_embedding(tails)).norm(p=self.norm, dim=1)
        return (heads+relations-tails).norm(p=self.norm, dim=1)

    def normalize(self, triplets: torch.LongTensor):
        heads = triplets[:, 0]
        s1 = (heads < self.offset).sum().item()
        s2 = (heads >= self.offset).sum().item()
        with torch.no_grad():
            if s2 == 0:
                # 都是a
                self.entities_emb_a.data = self.entities_emb_a.data / torch.norm(self.entities_emb_a, p=2, dim=1).view(
                    -1, 1)
            elif s1 == 0:
                # 都是b
                self.entities_emb_b.data = self.entities_emb_b.data / torch.norm(self.entities_emb_b, p=2, dim=1).view(
                    -1, 1)
            else:
                self.entities_emb_a.data = self.entities_emb_a.data / torch.norm(self.entities_emb_a, p=2, dim=1).view(
                    -1, 1)
                self.entities_emb_b.data = self.entities_emb_b.data / torch.norm(self.entities_emb_b, p=2, dim=1).view(
                    -1, 1)

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


def get_attr_corrupted(triples: torch.Tensor, entity_count1: int, entity_count2: int, attr_count1: int,
                       attr_count2: int, device: torch.device,
                       offset: int = 1000000):
    h = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]
    s1 = (h < offset).sum().item()
    s2 = (h >= offset).sum().item()
    head_or_tail = torch.randint(high=2, size=(h.size(0),), device=device)

    def func(entity_count, attr_count):
        random_entities = torch.randint(high=entity_count1, size=(h.size(0),), device=device)
        random_attrs = torch.randint(high=attr_count1, size=(h.size(0),), device=device)
        broken_heads = torch.where(head_or_tail == 1, random_entities, h)
        broken_tails = torch.where(head_or_tail == 0, random_attrs, t)
        return broken_heads, broken_tails

    with torch.no_grad():
        if s2 == 0:
            # 都是a
            random_entities = torch.randint(high=entity_count1, size=(h.size(0),), device=device)
            random_attrs = torch.randint(high=attr_count1, size=(h.size(0),), device=device)
            broken_heads = torch.where(head_or_tail == 1, random_entities, h)
            broken_tails = torch.where(head_or_tail == 0, random_attrs, t)
            return torch.stack([broken_heads, r, broken_tails], dim=1)
        elif s1 == 0:
            # 都是b
            random_entities = torch.randint(high=entity_count2, size=(h.size(0),), device=device)
            random_attrs = torch.randint(high=attr_count2, size=(h.size(0),), device=device)
            broken_heads = torch.where(head_or_tail == 1, random_entities, h)
            broken_tails = torch.where(head_or_tail == 0, random_attrs, t)
            return torch.stack([broken_heads, r, broken_tails], dim=1)
        else:
            random_entities_a = torch.randint(high=entity_count1, size=(h.size(0),), device=device)
            random_attrs_a = torch.randint(high=attr_count1, size=(h.size(0),), device=device)
            broken_heads_a = torch.where(head_or_tail == 1, random_entities_a, h)
            broken_tails_a = torch.where(head_or_tail == 0, random_attrs_a, t)

            random_entities_b = torch.randint(high=entity_count2, size=(h.size(0),), device=device)
            random_attrs_b = torch.randint(high=attr_count2, size=(h.size(0),), device=device)
            broken_heads_b = torch.where(head_or_tail == 1, random_entities_b, h)
            broken_tails_b = torch.where(head_or_tail == 0, random_attrs_b, t)

            broken_heads = torch.cat([broken_heads_a, broken_heads_b])