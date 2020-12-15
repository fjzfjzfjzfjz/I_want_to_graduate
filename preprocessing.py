# 划分训练集测试集
import pickle
import sys

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

# 获得字符编码集合
encoded_charset = char_set_fb | char_set_tw | char_set_fq
# char 2 id
encoded_charset = {i[1]: i[0] for i in enumerate(encoded_charset)}
pickle.dump(encoded_charset, open('preprocessed_data/encoded_charset.pickle', 'wb'))
sys.exit(0)
# 获得对齐的实体集合
aligned_fb_tw = []
aligned_fq_tw = []
aligned_fb_fq = []
# 因为只有twitter才保证有url，所以其他的都使用用户名标识当做url
# 所以制作url2id真
# fq根本没提供url
true_URL2id_fb = {userinfo_dict_fb[i]['URL']: i for i in userinfo_dict_fb}

# fq,tw
for i in userinfo_all_final_fq:
    idx = i[col_fq.index('number')]
    url_fb = i[col_fq.index('Facebook_url')]
    url_tw = i[col_fq.index('Twitter_url')]
    if url_fb.strip() != '' and url_fb in true_URL2id_fb:
        aligned_fb_fq.append([true_URL2id_fb[url_fb], idx])
    url_tw = url_tw.replace('http', 'https')
    if url_tw.strip() != '' and url_tw in URL2id_tw:
        aligned_fq_tw.append([idx, URL2id_tw[url_tw]])
for i in aligned_fq_tw:
    for j in aligned_fb_fq:
        if i[0] == j[1]:
            aligned_fb_tw.append([j[0], i[1]])
            break

pickle.dump(true_URL2id_fb, open('preprocessed_data/facebook/true_URL2id.pickle', 'wb'))
pickle.dump(aligned_fb_fq, open('preprocessed_data/aligned_fb_fq.pickle', 'wb'))
pickle.dump(aligned_fb_tw, open('preprocessed_data/aligned_fb_tw.pickle', 'wb'))
pickle.dump(aligned_fq_tw, open('preprocessed_data/aligned_fq_tw.pickle', 'wb'))

# 预处理出cor，即每一个rel 三元组的cor，否则速度太慢了
# 或者产生rel_dict，便于检查是否有关系
