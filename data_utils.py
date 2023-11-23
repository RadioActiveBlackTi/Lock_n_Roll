import pandas as pd
import numpy as np

def dataset_enroll(datapath, x, key_index, enroll_key=False):
    try:
        data = pd.read_csv(datapath)
    except:
        data = pd.DataFrame(columns=['Data','Key Index'])

    if not enroll_key:
        location = len(data)

        data.loc[location] = [x, key_index]
        data.to_csv(datapath, index=False)
    else:
        data.loc[data["Key Index"]==key_index] = [str(x), key_index]

        data.to_csv(datapath, index=False)

def dataset_extract(data, row_index, key_finding=False):
    if not key_finding:
        d = data.loc[row_index, "Data"]
        key = data.loc[row_index, "Key Index"]
    else:
        d = np.array(data.loc[data["Key Index"]==row_index, "Data"])[0]
        key = np.array(data.loc[data["Key Index"]==row_index, "Key Index"])[0]
    d = list(map(lambda x: x.replace('\n','').replace('[','').replace(']',''),d.split('[')))[2:]
    d = np.array(list(map(lambda x: np.fromstring(x, dtype=float, sep=' '), d)))

    return d, key

if __name__ == "__main__":
    dataset_enroll("./resources/test.csv", -4*np.ones((4,8)), 7)
    data = pd.read_csv("./resources/key.csv")
    d, key = dataset_extract(data,1, key_finding=True)
    print(d, d.shape)
    print(not not d.any())
    print(key)