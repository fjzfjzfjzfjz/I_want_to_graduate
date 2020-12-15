# 划分训练集测试集
import pickle

import numpy as np

col_tw = ['用户ID', 'URL', '推特里的userid', '昵称', '个人describe', '开通推特时间', 'location', 'tweet总数',
          'following总数', 'follower总数', 'favorites总数']
col_fq = ['number', 'id', 'username', 'Tips amount', 'Followers amount', 'Following amount', 'Facebook_url',
          'Twitter_url']
col_fb = ['id', 'URL', 'name', '昵称', 'Work', 'Education', 'current_city', 'hometown',
          'contact information', 'Relationship', 'Family Members', 'About',
          'Favorite Quotes', 'Gender', 'Interested In', 'Languages', 'Birthday']

char_set_fb = pickle.load(open('preprocessed_data/facebook/char_set.pickle', 'rb'))
userinfo_list_fb = pickle.load(open('preprocessed_data/facebook/userinfo_all_final.pickle', 'rb'))
userinfo_dict_fb = pickle.load(open('preprocessed_data/facebook/userinfo_dict.pickle', 'rb'))
userinfo_all_final_fb = pickle.load(open('preprocessed_data/facebook/userinfo_all_final.pickle', 'rb'))
URL2id_fb = pickle.load(open('preprocessed_data/facebook/URL2id.pickle', 'rb'))
id2URL_fb = pickle.load(open('preprocessed_data/facebook/id2URL.pickle', 'rb'))
rel_fb = pickle.load(open('preprocessed_data/facebook/rel.pickle', 'rb'))
attr_rel_fb = pickle.load(open('preprocessed_data/facebook/attr_rel.pickle', 'rb'))

char_set_fq = pickle.load(open('preprocessed_data/foursquare/char_set.pickle', 'rb'))
userinfo_all_final_fq = pickle.load(open('preprocessed_data/foursquare/userinfo_all_final.pickle', 'rb'))
URL2id_fq = pickle.load(open('preprocessed_data/foursquare/URL2id.pickle', 'rb'))
id2URL_fq = pickle.load(open('preprocessed_data/foursquare/id2URL.pickle', 'rb'))
rel_fq = pickle.load(open('preprocessed_data/foursquare/rel.pickle', 'rb'))
attr_rel_fq = pickle.load(open('preprocessed_data/foursquare/attr_rel.pickle', 'rb'))

char_set_tw = pickle.load(open('preprocessed_data/twitter/char_set.pickle', 'rb'))
userinfo_all_final_tw = pickle.load(open('preprocessed_data/twitter/userinfo_all_final.pickle', 'rb'))
URL2id_tw = pickle.load(open('preprocessed_data/twitter/URL2id.pickle', 'rb'))
id2URL_tw = pickle.load(open('preprocessed_data/twitter/id2URL.pickle', 'rb'))
rel_tw = pickle.load(open('preprocessed_data/twitter/rel.pickle', 'rb'))
attr_rel_tw = pickle.load(open('preprocessed_data/twitter/attr_rel.pickle', 'rb'))

encoded_charset = pickle.load(open('preprocessed_data/encoded_charset.pickle', 'rb'))

true_URL2id_fb = pickle.load(open('preprocessed_data/facebook/true_URL2id.pickle', 'rb'))
aligned_fb_fq = pickle.load(open('preprocessed_data/aligned_fb_fq.pickle', 'rb'))
aligned_fq_tw = pickle.load(open('preprocessed_data/aligned_fq_tw.pickle', 'rb'))
aligned_fb_tw = pickle.load(open('preprocessed_data/aligned_fb_tw.pickle', 'rb'))


# 使用数字作为rel，否则无法变成tensor
# rel_tw = [[int(i[0]), int(i[1])] for i in rel_tw]
# rel_fq = [[int(i[0]), int(i[1])] for i in rel_fq]
# rel_fb = [[int(i[0]), int(i[1])] for i in rel_fb]
# pickle.dump(rel_tw, open('preprocessed_data/twitter/rel.pickle', 'wb'))
# pickle.dump(rel_fq, open('preprocessed_data/foursquare/rel.pickle', 'wb'))
# pickle.dump(rel_fb, open('preprocessed_data/facebook/rel.pickle', 'wb'))

# 发现属性有空字符，进行去除
# attr_rel_tw = [i for i in attr_rel_tw if i[-1].strip() != '']
# attr_rel_fq = [i for i in attr_rel_fq if i[-1].strip() != '']
# attr_rel_fb = [i for i in attr_rel_fb if i[-1].strip() != '']
# pickle.dump(attr_rel_tw, open('preprocessed_data/twitter/attr_rel.pickle', 'wb'))
# pickle.dump(attr_rel_fq, open('preprocessed_data/foursquare/attr_rel.pickle', 'wb'))
# pickle.dump(attr_rel_fb, open('preprocessed_data/facebook/attr_rel.pickle', 'wb'))
# 使用数字作为rel，否则无法变成tensor
# attr_rel_tw = [[int(i[0]), int(i[1]), i[2]] for i in attr_rel_tw]
# attr_rel_fq = [[int(i[0]), int(i[1]), i[2]] for i in attr_rel_fq]
# attr_rel_fb = [[int(i[0]), int(i[1]), i[2]] for i in attr_rel_fb]
# pickle.dump(attr_rel_tw, open('preprocessed_data/twitter/attr_rel.pickle', 'wb'))
# pickle.dump(attr_rel_fq, open('preprocessed_data/foursquare/attr_rel.pickle', 'wb'))
# pickle.dump(attr_rel_fb, open('preprocessed_data/facebook/attr_rel.pickle', 'wb'))

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

rel_tw_cor = pickle.load(open('preprocessed_data/twitter/rel_tw_cor.pickle', 'rb'))
rel_fq_cor = pickle.load(open('preprocessed_data/foursquare/rel_fq_cor.pickle', 'rb'))
rel_fb_cor = pickle.load(open('preprocessed_data/facebook/rel_fb_cor.pickle', 'rb'))
# 或者产生rel_dict，便于检查是否有关系
# def get_rel_dict(tt):
#     rt = {}
#     for i in tt:
#         if i[0] not in rt:
#             rt[i[0]] = {i[1]: 1}
#         else:
#             rt[i[0]][i[1]] = 1
#     return rt
#
#
# rel_tw_dict = get_rel_dict(rel_tw)
# rel_fq_dict = get_rel_dict(rel_fq)
# rel_fb_dict = get_rel_dict(rel_fb)
