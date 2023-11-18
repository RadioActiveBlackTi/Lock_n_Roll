import pandas as pd
import numpy as np

def dataset_enroll(datapath, x, key_index):
    try:
        data = pd.read_csv(datapath)
    except:
        data = pd.DataFrame(columns=['Data','Key Index'])

    location = len(data)

    data.loc[location]=[x, key_index]
    data.to_csv(datapath, index=False)

def dataset_extract(data, row_index):
    d = data.loc[row_index, "Data"]
    d = list(map(lambda x: x.replace('\n','').replace('[','').replace(']',''),d.split('[')))[2:]
    d = np.array(list(map(lambda x: np.fromstring(x, dtype=float, sep=' '), d)))
    key = data.loc[row_index, "Key Index"]

    return d, key

if __name__ == "__main__":
    dataset_enroll("./test.csv", -4*np.ones((4,8)), 7)
    data = pd.read_csv("./dataset.csv")
    d, key = dataset_extract(data,0)
    print(d, d.shape)
    print(key)