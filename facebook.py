# 将friends following整理成文件
import os
import pickle

output_path = 'data/facebook/followings'
lines = {}
userinfo_dict = {}
userinfo_list = []
char_set = set()

col = ['id', 'URL', 'name', '昵称', 'Work', 'Education', 'current_city', 'hometown',
       'contact information', 'Relationship', 'Family Members', 'About',
       'Favorite Quotes', 'Gender', 'Interested In', 'Languages', 'Birthday']
dont_need = ['#education', '#{#', '#}#', '#living', '#relationship',
             '#bio', '#year-overviews', 'total']
useless = ['@Other Places Lived', '@Life Events']
rt_dict = {'@Education': 3, '@Work': 2, '@Current City and Hometown:': 4, 'Current city': 5, 'Hometown': 6,
           '@Contact Information': 6, '@Basic Information': 11, '@Relationship': 7,
           '@Family Members': 8, '@About': 9, '@Favorite Quotes': 10, 'Interested In': 12,
           'Languages': 13, 'Gender': 11, 'Birthday': 14}


def check_has(x: list, y: str):
    for i in x:
        if i in y:
            return True
    return False


def parse(text):
    # print(text)
    # 将每个用户的组合起来，解析出id:[url。。。]
    for k in text:
        try:
            idx = int(k[:k.find(':')])
            idx=str(idx)
            # 开始行
            # assert idx not in lines,"重复处理关系 "+str(idx)
            # TODO 使用entity的编号区分吧，entity重复就选一个
            assert k.count(':') >= 3
            # https://也有冒号，所以不能split
            lines[idx] = []
        except ValueError:
            # 不是开始行,不处理assert
            if 'total' in k or '#{#' in k or '#}#' in k:
                continue
            assert k.count('#') >= 2, k
            lines[idx].append(k.split('#')[0].strip())
    return lines


# 把分隔符视为一样，不区分@Work和没有@等。。
def update(this_part, k):
    if check_has(useless, k):
        return -1
    for i in rt_dict:
        if i in k:
            return rt_dict[i] + 2  # 因为少写了url 和 name。。。


def parse_userinfo(text):
    # print(text)
    # 解析出userinfo {id:{}} and [xxx]
    for k in text:
        try:
            if k.count(':') < 3:
                raise ValueError
            idx = int(k[:k.find(':')])
            idx = str(idx)
            d = k.strip().split(':')
            # print(d)
            username = d[2].strip() if d[2].strip() != '' else d[1].strip()
            uu = d[1].strip()
            # 开始行
            # https://也有冒号，所以需要组合url
            url = ':'.join(d[3:])
            userinfo_dict[idx] = {'昵称': username, 'URL': url, 'name': uu}
            for l in col[4:]:
                userinfo_dict[idx][l] = ''
            # user_info_dict[idx] = {'昵称': username, 'URL': url, 'name': uu}
            userinfo_list.append([idx, url, uu, username] + [''] * (len(col) - 3))
            this_part = 2  # 当前处理的子索引，即Education

        except ValueError:
            # 不是开始行,不处理assert
            if check_has(dont_need, k):
                continue
            if check_has(list(rt_dict.keys()) + useless, k):
                # 更新所在子索引
                this_part = update(this_part, k)
            else:
                # 输入数据
                try:
                    if this_part == -1: continue  # 不读取的数据
                    if 'to show' in k: continue
                    userinfo_list[-1][this_part] += k.strip()
                    userinfo_list[-1][this_part] += ' '
                    userinfo_dict[idx][col[this_part]] += k.strip()
                    userinfo_dict[idx][col[this_part]] += ' '
                except UnboundLocalError:
                    print(k)
                    raise Exception


for path in os.listdir('data/facebook'):
    # print(path)
    if not os.path.isdir(f'data/facebook/{path}') or 'following' in path: continue
    dd = os.listdir(f'data/facebook/{path}')
    name = ''
    # print(path)
    for i in dd:
        if 'friend' in i:
            name = i
            break
    assert name != ''
    # print(name)
    # print(f'data/facebook/{path}/{name}')
    for file_name in os.listdir(f'data/facebook/{path}/{name}'):
        if check_has(['log', 'wronglink'], file_name):
            continue
        # print(f'data/facebook/{path}/{name}/{file_name}')
        with open(f'data/facebook/{path}/{name}/{file_name}', encoding='utf-8-sig') as f:
            parse(f.readlines())
    # 提取所有的字符，简便处理直接从文件中读取
    name = ''
    for i in dd:
        if 'about' in i:
            name = i
            break
    assert name != ''
    # print(f'data/facebook/{path}/{name}')
    for file_name in os.listdir(f'data/facebook/{path}/{name}'):
        if check_has(['log', 'wronglink'], file_name):
            continue
        print(f'data/facebook/{path}/{name}/{file_name}')
        with open(f'data/facebook/{path}/{name}/{file_name}', encoding='utf-8-sig') as f:
            for s in f:
                char_set |= set(s.strip())
            # 获得userinfo信息，直接读取为dict，不存成txt
        with open(f'data/facebook/{path}/{name}/{file_name}', encoding='utf-8-sig') as f:
            parse_userinfo(f.readlines())

# 使用name表示URL
URL2id = {userinfo_dict[i]['name']: i for i in userinfo_dict}
id2URL = {i: userinfo_dict[i]['name'] for i in userinfo_dict}
# 去除无效（无name）实体
d = []
for i in userinfo_dict:
    if userinfo_dict[i]['name'].strip() == '':
        d.append(i)
for i in d:
    del userinfo_dict[i]
# 去除无意义的联系
rel = []
tt = 0
nc={userinfo_dict[i]['昵称'] for i in userinfo_dict}
for i in lines:
    if i not in id2URL:
        continue
    for j in lines[i]:
        if j not in URL2id and j not in nc:
            tt += 1
        else:
            rel.append([str(i), URL2id[j]])
print(f'无效记录 {tt}')

# 属性三元组（h，r，属性值）
attr_rel = []
encoding = {'昵称': 0, 'Work': 9, 'Education': 10, 'current_city': 11, 'hometown': 12,
            'contact information': 13, 'Relationship': 14, 'Family Members': 15, 'About': 16,
            'Favorite Quotes': 17, 'Gender': 18, 'Interested In': 19, 'Languages': 20, 'Birthday': 21}
attrs = ['昵称', 'Work', 'Education', 'current_city', 'hometown',
         'contact information', 'Relationship', 'Family Members', 'About',
         'Favorite Quotes', 'Gender', 'Interested In', 'Languages', 'Birthday']
for idx in userinfo_dict:
    for j in attrs:
        attr_rel.append([idx, encoding[j], userinfo_dict[idx][j]])

# save
with open('data/facebook/char_set.txt', 'w', encoding='utf-8') as f:
    f.writelines([i + '\n' for i in char_set])
pickle.dump(char_set, open('data/facebook/char_set.pickle', 'wb'))
pickle.dump(userinfo_list, open('data/facebook/userinfo_all_final.pickle', 'wb'))
pickle.dump(userinfo_dict, open('data/facebook/userinfo_dict.pickle', 'wb'))
pickle.dump(URL2id, open('data/facebook/URL2id.pickle', 'wb'))
pickle.dump(id2URL, open('data/facebook/id2URL.pickle', 'wb'))
with open('data/facebook/rel.txt', 'w', encoding='utf-8') as f:
    f.writelines([' '.join(i) for i in rel])
pickle.dump(rel, open('data/facebook/rel.pickle', 'wb'))
pickle.dump(attr_rel, open('data/facebook/attr_rel.pickle', 'wb'))
