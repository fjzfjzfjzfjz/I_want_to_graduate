# # 对tw，rel进行重新编号
# def re_index_with_str_key(id2URL, URL2id, rel):
#     mapping = {}
#     keys = [int(i) for i in id2URL]
#     keys.sort()
#     ii = 0
#     for i in keys:
#         if i not in mapping:
#             mapping[i] = ii
#             ii += 1
#     r1 = {mapping[i]: id2URL[str(i)] for i in keys}
#     r2 = {i: mapping[int(URL2id[i])] for i in URL2id}
#     r3 = [[mapping[i[0]], mapping[i[1]]] for i in rel]
#     return r1, r2, r3
#
#
# def re_index_with_int_key(id2URL, URL2id, rel):
#     mapping = {}
#     keys = list(id2URL)
#     keys.sort()
#     ii = 0
#     for i in keys:
#         if i not in mapping:
#             mapping[i] = ii
#             ii += 1
#     r1 = {mapping[i]: id2URL[i] for i in keys}
#     r2 = {i: mapping[int(URL2id[i])] for i in URL2id}
#     r3 = [[mapping[i[0]], i[1], mapping[i[2]]] for i in rel]
#     return r1, r2, r3
#
#
# def get_data(id2URL, URL2id, rel):
#     id2URL, URL2id, rel = re_index_with_str_key(id2URL, URL2id, rel)
#     rel = [[i[0], i[0], i[1]] for i in rel]
#     # rel = [[i[0], 0, i[1]] for i in rel]
#     cd = np.array(rel)
#     cc = Counter(np.concatenate([cd[:, 0], cd[:, 2]]))
#     rt = []
#     for i in cc:
#         if cc[i] == 0:
#             rt.append(i)
#     id2URL = {i: id2URL[i] for i in id2URL if int(i) in cc and cc[int(i)] != 0}
#     URL2id = {id2URL[i]: i for i in id2URL}
#     id2URL, URL2id, rel = re_index_with_int_key(id2URL, URL2id, rel)
#     rel = [[i[0], i[0], i[2]] for i in rel]
#     train1 = []
#     test1 = []
#     for i in rel:
#         c = np.random.random()
#         if c < 0.8:
#             train1.append(i)
#         else:
#             test1.append(i)
#     return train1, test1

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

# 对tw，rel进行重新编号
# def re_index(id2URL, URL2id, rel):
#     mapping = {}
#     keys = [int(i) for i in id2URL]
#     keys.sort()
#     ii = 0
#     for i in keys:
#         if i not in mapping:
#             mapping[i] = ii
#             ii += 1
#
#     r1 = {mapping[i]: id2URL[str(i)] for i in keys}
#     r2 = {i: mapping[int(URL2id[i])] for i in URL2id}
#     r3 = [[mapping[i[0]], mapping[i[1]]] for i in rel]
#     return r1, r2, r3
#
#
# id2URL_tw, URL2id_tw, rel_tw = re_index(id2URL_tw, URL2id_tw, rel_tw)



