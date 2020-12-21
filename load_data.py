# 划分训练集测试集
import pickle

import numpy as np

"""
rel,URL2id,id2URL 的id均使用int，并扩展关注关系，并去除无关注或被关注的实体，后进行重新编号
xxx_final.pickle 即处理后的版本
xxx_expanded.pickle 即扩展的版本
xxx.pickle没有去除无效实体的版本，但是进行了重新编号
xxx.ori没有进行重新编号
因为考虑到论文编写的需要，所以pickle包括 普通；扩展关注；扩展关注并去除无关注或被关注的的实体，三种。均为int id、均重新编号
处理代码在处理代码中，很乱
"""
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
URL2id_fb = pickle.load(open('preprocessed_data/facebook/URL2id_final.pickle', 'rb'))
id2URL_fb = pickle.load(open('preprocessed_data/facebook/id2URL_final.pickle', 'rb'))
rel_fb = pickle.load(open('preprocessed_data/facebook/rel_final.pickle', 'rb'))
attr_rel_fb = pickle.load(open('preprocessed_data/facebook/attr_rel.pickle', 'rb'))

char_set_fq = pickle.load(open('preprocessed_data/foursquare/char_set.pickle', 'rb'))
userinfo_all_final_fq = pickle.load(open('preprocessed_data/foursquare/userinfo_all_final.pickle', 'rb'))
URL2id_fq = pickle.load(open('preprocessed_data/foursquare/URL2id_final.pickle', 'rb'))
id2URL_fq = pickle.load(open('preprocessed_data/foursquare/id2URL_final.pickle', 'rb'))
rel_fq = pickle.load(open('preprocessed_data/foursquare/rel_final.pickle', 'rb'))
attr_rel_fq = pickle.load(open('preprocessed_data/foursquare/attr_rel.pickle', 'rb'))

char_set_tw = pickle.load(open('preprocessed_data/twitter/char_set.pickle', 'rb'))
userinfo_all_final_tw = pickle.load(open('preprocessed_data/twitter/userinfo_all_final.pickle', 'rb'))
URL2id_tw = pickle.load(open('preprocessed_data/twitter/URL2id_final.pickle', 'rb'))
id2URL_tw = pickle.load(open('preprocessed_data/twitter/id2URL_final.pickle', 'rb'))
rel_tw = pickle.load(open('preprocessed_data/twitter/rel_final.pickle', 'rb'))
attr_rel_tw = pickle.load(open('preprocessed_data/twitter/attr_rel.pickle', 'rb'))

encoded_charset = pickle.load(open('preprocessed_data/encoded_charset.pickle', 'rb'))

true_URL2id_fb = pickle.load(open('preprocessed_data/facebook/true_URL2id.pickle', 'rb'))
aligned_fb_fq = pickle.load(open('preprocessed_data/aligned_fb_fq.pickle', 'rb'))
aligned_fq_tw = pickle.load(open('preprocessed_data/aligned_fq_tw.pickle', 'rb'))
aligned_fb_tw = pickle.load(open('preprocessed_data/aligned_fb_tw.pickle', 'rb'))


def split_data(data):
    data = np.array(data)
    idxs = np.random.randint(0, 10, size=(len(data),))
    train1 = data[idxs < 8, :].tolist()
    test1 = data[idxs >= 8, :].tolist()
    return train1, test1


train_tw, test_tw = split_data(rel_tw)
train_fq, test_fq = split_data(rel_fq)
train_fb, test_fb = split_data(rel_fb)


# 获得关系出边的比例，即alpha
# def get(rel, id2URL):
#     cc = Counter([i[1] for i in rel])
#     rt = {i: cc[i] / len(rel) for i in id2URL}
#     return rt
#
#
# alpha_tw = get(rel_tw, id2URL_tw)
# alpha_fb = get(rel_fb, id2URL_fb)
# alpha_fq = get(rel_fq, id2URL_fq)


def prepare(data, ll):
    return data[:ll] + [0] * (ll - len(data))


# 将属性字符串全部变成字符的索引集合
# 对attr_rel_tw进行id编码
def change(attr_rel):
    rt = []
    attrs = []
    for i in attr_rel:
        t = [encoded_charset[j] for j in i[-1]]
        t = prepare(t, 160)
        rt.append([i[0], i[1], len(attrs)])
        attrs.append(t)
    return rt, attrs


attr_rel_tw, attrs_only_tw = change(attr_rel_tw)
attr_rel_fq, attrs_only_fq = change(attr_rel_fq)
attr_rel_fb, attrs_only_fb = change(attr_rel_fb)


def cccount(data):
    s = set()
    for i in data:
        s.add(i[1])
    return len(s)


attr_rel_tw_count = 8
attr_rel_fq_count = 4
attr_rel_fb_count = 14
merged_tw_fq_attr_rel_count = 9
merged_tw_fb_attr_rel_count = 21
merged_fb_fq_attr_rel_count = 17


# attr_rel_tw_count = cccount(attr_rel_tw)
# attr_rel_fq_count = cccount(attr_rel_fq)
# attr_rel_fb_count = cccount(attr_rel_fb)
# merged_tw_fq_attr_rel_count = cccount(attr_rel_tw + attr_rel_fq)
# merged_tw_fb_attr_rel_count = cccount(attr_rel_tw + attr_rel_fb)
# merged_fb_fq_attr_rel_count = cccount(attr_rel_fb + attr_rel_fq)


# len(list(filter(lambda x: x <= 100, [len(i) for i in attrs_only_fb])))
# c = len(list(filter(lambda x: x > 10000, [len(i) for i in attrs_only_fb])))
# a = len(attrs_only_fb) + len(attrs_only_tw) + len(attrs_only_fq)
# print((a - c) / a * 100)
# 先规定为160，进行tw fq测试,tw 160;fq 46;fb 29472

# def prepare_data(tw=False, fb=False, fq=False):
#     assert tw + fb + fq == 2
#     ltw = max([len(i) for i in attrs_only_tw])
#     lfb = max([len(i) for i in attrs_only_fb])
#     lfq = max([len(i) for i in attrs_only_fq])
#     if fb is False:
#         ll = max(ltw, lfq)
#         a=attrs_only_tw
#         b=attrs_only_fq

train_attr_tw, test_attr_tw = split_data(attr_rel_tw)
train_attr_fq, test_attr_fq = split_data(attr_rel_fq)
train_attr_fb, test_attr_fb = split_data(attr_rel_fb)
