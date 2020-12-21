import numpy as np
from torch.utils.data import Dataset


class RelationDataset(Dataset):
    def __init__(self, data):
        super(RelationDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestRelationDataset(Dataset):
    def __init__(self, data, test_samples=5000):
        super(TestRelationDataset, self).__init__()
        if test_samples is None:
            self.data = data
        else:
            dd = np.array(data)
            cc = np.random.choice(range(dd.shape[0]), (test_samples,))
            self.data = dd[cc, :].tolist()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class AttributeRelationDataset(Dataset):
    def __init__(self, data, attrs):
        super(AttributeRelationDataset, self).__init__()
        assert len(data) == len(attrs)
        self.data = data
        self.attrs = attrs

    def __getitem__(self, index):
        return self.data[index][:-1], self.attrs[index]

    def __len__(self):
        return len(self.data)
