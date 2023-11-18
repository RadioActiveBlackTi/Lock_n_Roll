import torch
import pandas as pd
import numpy as np
from data_utils import dataset_extract
from torch.utils.data import Dataset

class PairwiseDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)

        inp = []
        outp = []
        data_length = len(df)
        for i in range(data_length):
            for j in range(data_length):
                d_a, key_a = dataset_extract(df, i)
                d_b, key_b = dataset_extract(df, j)
                if key_a==key_b:
                    outp.append(1)
                    inp.append((d_a, d_b))
                else:
                    outp.append(0)
                    inp.append((d_a, d_b))
                    """
                    k = random.choice([abc for abc in range(data_length**2)])
                    if (k < 3*data_length):
                        outp.append(0)
                        inp.append((d_a, d_b))
                    """
        self.inp = inp
        self.outp = np.asarray(outp)

    def __len__(self):
        return len(self.outp)

    def __getitem__(self,idx):
        inp = (torch.FloatTensor(self.inp[idx][0]), torch.FloatTensor(self.inp[idx][1]))
        outp = torch.FloatTensor([self.outp[idx]])
        return inp, outp

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from collections import Counter

    dataset = PairwiseDataset("./dataset.csv")
    counts = dict(Counter(dataset.outp))
    print(counts)
    weights = torch.FloatTensor([1/counts[0], 1/counts[1]])
    weights = weights / torch.sum(weights)

    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(1): # Since pairwise, you have to split train data into two pieces
        for batch_idx, samples in enumerate(dataloader):
            train_data, label_train = samples # train_data: tuple(Tensor(B, T, L), Tensor(B, T, L))
            x_train, y_train = train_data[0], train_data[1]
            print(x_train.shape)
            print(label_train)

