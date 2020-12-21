import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    entity, relation, G = {}, {}, []
    with open('FB15k/entity2id.txt', 'r') as f:
        for line in f:
            temp = line.replace(' ', '\t').strip().split('\t')
            assert len(temp) >= 2, 'decode ERROR'
            temp = list(map(str.strip, temp))
            entity[temp[0]] = temp[1]
    with open('FB15k/relation2id.txt', 'r') as f:
        for line in f:
            temp = line.replace(' ', '\t').strip().split('\t')
            assert len(temp) >= 2, 'decode ERROR'
            temp = list(map(str.strip, temp))
            relation[temp[0]] = temp[1]
    with open('FB15k/train.txt', 'r') as f:
        for line in f:
            temp = line.replace(' ', '\t').strip().split('\t')
            assert len(temp) >= 3, 'decode ERROR'
            temp = list(map(str.strip, temp))
            row = [entity[temp[0]], relation[temp[2]], entity[temp[1]]]
            G.append(row)
    return entity, relation, G


def load_data1():
    entity, relation, G = {}, {}, []
    with open('FB15k/entity2id.txt', 'r') as f:
        for line in f:
            temp = line.replace(' ', '\t').strip().split('\t')
            assert len(temp) >= 2, 'decode ERROR'
            temp = list(map(str.strip, temp))
            entity[temp[0]] = temp[1]
    with open('FB15k/relation2id.txt', 'r') as f:
        for line in f:
            temp = line.replace(' ', '\t').strip().split('\t')
            assert len(temp) >= 2, 'decode ERROR'
            temp = list(map(str.strip, temp))
            relation[temp[0]] = temp[1]
    with open('FB15k/test.txt', 'r') as f:
        for line in f:
            temp = line.replace(' ', '\t').strip().split('\t')
            assert len(temp) >= 3, 'decode ERROR'
            temp = list(map(str.strip, temp))
            row = [entity[temp[0]], relation[temp[2]], entity[temp[1]]]
            G.append(row)
    return entity, relation, G


entity2id, relation2id, rel = load_data()
_, _, test1 = load_data1()
test1 = [[int(i[0]), int(i[1]), int(i[2])] for i in test1]
rel = [[int(i[0]), int(i[1]), int(i[2])] for i in rel]
test1 = torch.Tensor(test1).long().to(device)
