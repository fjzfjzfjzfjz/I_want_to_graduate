from torch.utils.data import Dataset


class RelDataset(Dataset):
    def __init__(self, data):
        super(RelDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


