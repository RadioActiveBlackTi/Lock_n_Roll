import torch.nn as nn
import torch
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(num_layers*hidden_size*2, hidden_size)
        self.fc_concat = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_final = nn.Linear(hidden_size, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, y):
        # x, y: (B, T, L)
        B, L = x.shape[0], x.shape[1]

        h_0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)).to(device)
        _, (x_encode, _) = self.lstm(x, (h_0, c_0))
        _, (y_encode, _) = self.lstm(y, (h_0, c_0))

        x_encode = x_encode.transpose(0,1).transpose(1,2).reshape(B,-1)
        y_encode = y_encode.transpose(0,1).transpose(1,2).reshape(B,-1)

        x_encode = self.fc1(x_encode)
        x_encode = self.bn1(x_encode)
        x_encode = self.dropout1(x_encode)
        x_encode = self.leaky_relu(x_encode)

        y_encode = self.fc1(y_encode)
        y_encode = self.bn1(y_encode)
        y_encode = self.dropout1(y_encode)
        y_encode = self.leaky_relu(y_encode)

        out = torch.cat([x_encode, y_encode], dim=1)
        out = self.fc_concat(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        out = self.relu(out)
        out = self.fc_final(out)
        out = torch.sigmoid(out)

        return out

if __name__ == "__main__":
    x = y = torch.ones([3, 4, 8])
    d = Discriminator(8, 64, 5)
    print(d(x,y))