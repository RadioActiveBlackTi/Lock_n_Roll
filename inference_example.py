import torch
from model import Discriminator
from data_utils import dataset_extract
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__=="__main__":
    net = Discriminator(8, 128, 8)
    net.load_state_dict(torch.load("./resources/model_state_dict.pt", map_location=torch.device(device)))
    net.to(device)

    data = pd.read_csv("./resources/dataset.csv")

    x, _ = dataset_extract(data, 1)
    y, _ = dataset_extract(data, 8)
    z, _ = dataset_extract(data, 11)

    d = """[[ 7.5   9.1   7.58 10.73  9.17  0.49 -0.17 49.02]
    [ 7.4   7.9   7.47 10.65  9.14  1.34 -0.18 52.76]
    [10.2   7.7   7.39 10.55  9.16  1.88 -0.13 60.38]
    [ 6.6   8.1   7.05 10.29  9.01  1.79 -0.42 43.43]
    [ 6.5   8.    6.85 10.14  8.88  1.27 -0.73 43.35]
    [ 9.1   8.8   7.19 10.5   9.19  1.35  0.13 43.44]
    [27.1   8.    7.33 10.55  9.22  0.9   0.19 43.2 ]
    [12.3   8.3   7.17 10.29  8.99  0.64 -3.61 47.15]
    [12.7   6.    7.19 10.36  8.99  0.71 -0.47 45.12]]"""
    d = list(map(lambda x: x.replace('\n', '').replace('[', '').replace(']', ''), d.split('[')))[2:]
    d = np.array(list(map(lambda x: np.fromstring(x, dtype=float, sep=' '), d)))

    t = """[[ 7.6   8.9   7.6  10.81  9.25  1.92 -0.54 45.62]
    [ 7.4   8.    7.51 10.72  9.23  0.88  0.27 51.77]
    [11.    7.6   7.33 10.52  9.13  0.22  2.4  50.73]
    [ 6.6   8.3   6.95 10.14  8.89 -0.36  1.66 50.78]
    [ 6.4   8.1   6.77  9.93  8.71  1.6  -1.31 -0.38]
    [ 9.2   8.6   7.06 10.31  9.02  1.84 -0.49 11.39]
    [21.9   7.8   7.18 10.34  9.06  0.6  -6.52  9.98]
    [11.3   9.1   7.07 10.15  8.88 -0.31  0.45  8.54]
    [12.2   6.    7.06 10.2   8.86  1.06 -0.23 62.7 ]]"""
    t = list(map(lambda x: x.replace('\n', '').replace('[', '').replace(']', ''), t.split('[')))[2:]
    t = np.array(list(map(lambda x: np.fromstring(x, dtype=float, sep=' '), t)))

    x = torch.FloatTensor(x).unsqueeze(0).to(device)
    y = torch.FloatTensor(y).unsqueeze(0).to(device)
    z = torch.FloatTensor(z).unsqueeze(0).to(device)
    d, t = torch.FloatTensor(d).unsqueeze(0).to(device), torch.FloatTensor(t).unsqueeze(0).to(device)

    print(d, t)

    net.eval()
    print(x, y, z)
    print(y.shape)

    print('\nposition change test')
    print(net(x, y))
    print(net(y, x))

    print('\ndifferent test')
    print(net(z, y))
    print(net(x, z))

    print('\nsame test')
    print(net(x, x))
    print(net(y, y))

    print('\nlinear transformed data test')
    print(net(2 * x + 6, 2 * x))
    print(net(15 * y, 5 * x + 3))

    print('\nreal same data test')
    print(net(d, t))
    print(net(t, d))

    print('\nreal different data test')
    print(net(d, x))
    print(net(y, t))
    print(net(d, z))
    print(net(y, d))