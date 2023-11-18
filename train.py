import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import Accuracy, BinaryPrecision, BinaryRecall
from collections import Counter

from model import Discriminator
from pairwise_dataset import PairwiseDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, epochs, loss_weights, dataloader_train, dataloader_valid):
    bce_loss = nn.BCELoss(reduction='none').to(device)
    criterion = lambda pred, target: ((loss_weights[1] * target + loss_weights[0] * (1 - target)) * bce_loss(pred, target)).mean()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    metric = Accuracy(task="binary")
    metric2 = BinaryPrecision()
    metric3 = BinaryRecall()

    for epoch in range(epochs):
        cost = 0.0

        model.train()
        metric.reset()

        for batch_idx, samples in enumerate(dataloader_train):
            train_data, label_train = samples # train_data: tuple(Tensor(B, T, L), Tensor(B, T, L))
            x_train, y_train = train_data[0], train_data[1]

            x_train.to(device)
            y_train.to(device)
            label_train.to(device)

            optimizer.zero_grad()

            output = model(x_train, y_train)
            loss = criterion(output, label_train)

            loss.backward()
            optimizer.step()

            cost += loss
            metric(output, label_train)
            metric2(output, label_train)
            metric3(output, label_train)

        cost = cost / len(dataloader_train)
        acc = metric.compute()
        prc = metric2.compute()
        rca = metric3.compute()
        print(f"Epoch {epoch} - Train Cost: {cost}     Train Accuracy: {acc}     Train Precision: {prc}     Train Recall: {rca}")

        metric.reset()
        metric2.reset()
        metric3.reset()

        val_cost = 0.0
        with torch.no_grad():
            model.eval()
            for batch_idx, samples in enumerate(dataloader_valid):
                valid_data, label_valid = samples
                x, y = valid_data[0], valid_data[1]

                x.to(device)
                y.to(device)
                label_valid.to(device)

                optimizer.zero_grad()

                output = model(x, y)
                loss = criterion(output, label_valid)

                val_cost += loss
                metric(output, label_valid)
                metric2(output, label_valid)
                metric3(output, label_valid)

            val_cost /= len(dataloader_valid)
            acc = metric.compute()
            prc = metric2.compute()
            rca = metric3.compute()
            print(f"Epoch {epoch} - Valid Cost: {val_cost}     Valid Accuracy: {acc}     Valid Precision: {prc}     Valid Recall: {rca}")

def test(model, loss_weights, dataloader_test):
    test_cost = 0.0

    bce_loss = nn.BCELoss(reduction='none').to(device)
    criterion = lambda pred, target: ((loss_weights[1] * target + loss_weights[0] * (1 - target)) * bce_loss(pred, target)).mean()
    metric = Accuracy(task="binary")
    metric2 = BinaryPrecision()
    metric3 = BinaryRecall()

    with torch.no_grad():
        model.eval()
        for batch_idx, samples in enumerate(dataloader_test):
            test_data, test_label = samples
            x, y = test_data[0], test_data[1]

            x.to(device)
            y.to(device)
            test_label.to(device)

            output = model(x, y)
            loss = criterion(output, test_label)

            test_cost += loss
            metric(output, test_label)
            metric2(output, test_label)
            metric3(output, test_label)

        test_cost /= len(dataloader_valid)
        acc = metric.compute()
        prc = metric2.compute()
        rca = metric3.compute()
        print(f"Test - Test Cost: {test_cost}     Test Accuracy: {acc}     Test Precision: {prc}     Test Recall: {rca}")

if __name__=="__main__":
    epochs = 50
    dataset = PairwiseDataset("./dataset.csv")

    counts = dict(Counter(dataset.outp))
    print(counts)
    weights = torch.FloatTensor([1 / counts[0], 1 / counts[1]])
    weights = weights / torch.sum(weights)
    print(weights)

    dataset_size = len(dataset)
    train_size = int(dataset_size*0.7)
    validation_size = int(dataset_size*0.2)
    test_size = dataset_size - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    dataloader_valid = DataLoader(validation_dataset, batch_size=64, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(validation_dataset, batch_size=64, shuffle=True, drop_last=True)

    net = Discriminator(8, 128, 8).to(device)
    print(device)
    train(net, epochs, weights, dataloader_train, dataloader_valid)
    print("="*32)
    test(net, weights, dataloader_test)
    if ((input("Save it?: "))!="n"):
        torch.save(net.state_dict(), './model_state_dict.pt')
        print('saved.')
