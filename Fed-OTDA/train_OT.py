import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from geomloss import SamplesLoss
import torch.nn as nn
import torch
import torch.utils.data as util_data
from test import *
import pandas as pd


# Load datasets
Source_train = pd.read_csv("/content/drive/MyDrive/OTRUDA/data/Source_train.csv")
Source_test = pd.read_csv("/content/drive/MyDrive/OTRUDA/data/Source_test.csv")
Target_train = pd.read_csv("/content/drive/MyDrive/OTRUDA/data/Target_train.csv")
Target_test = pd.read_csv("/content/drive/MyDrive/OTRUDA/data/Target_test.csv")

FEATURES_dset = [i for i in Source_train.columns if i != 'labels']
len_features = len(FEATURES_dset)
unique_labels = Source_train['labels'].unique().tolist()
print('Number of labels:', len(unique_labels))

# Dataset class
class PytorchDataSet(util_data.Dataset):
    def __init__(self, df, len_features):
        FEATURES = [i for i in df.columns if i != 'labels']
        TARGET = "labels"

        from sklearn.preprocessing import StandardScaler
        Normarizescaler = StandardScaler()
        Normarizescaler.fit(np.array(df[FEATURES]))

        if "labels" not in df.columns:
            df["labels"] = 9999

        self.df = df
        self.train_X = Normarizescaler.transform(np.array(self.df[FEATURES]))
        self.train_Y = np.array(self.df[TARGET])

        self.train_X = torch.from_numpy(self.train_X).float()
        self.train_Y = torch.from_numpy(self.train_Y).long()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.train_X[idx].view(1, len_features), self.train_Y[idx]


# Model definition
class ResNet1D(nn.Module):
    def __init__(self, num_classes=len(unique_labels)):
        super(ResNet1D, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=1, stride=1)
        self.layer1 = self._make_custom_layer(64, 64, 3)
        self.layer2 = self._make_custom_layer(64, 128, 4, stride=1)
        self.layer3 = self._make_custom_layer(128, 256, 6, stride=1)
        self.layer4 = self._make_custom_layer(256, 512, 3, stride=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_custom_layer(self, inplanes, planes, blocks, stride=1):
        layers = [self._make_block(inplanes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(self._make_block(planes, planes))
        return nn.Sequential(*layers)

    def _make_block(self, inplanes, planes, stride=1):
        return nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def print_model_layers(model):
    print("Model architecture:\n")
    for name, layer in model.named_children():
        print(f"{name}: {layer}")
        if isinstance(layer, nn.Sequential):
            for sub_name, sub_layer in layer.named_children():
                print(f"  {sub_name}: {sub_layer}")
        print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--Lambda', type=float, default=0.1, help='lambda value for OT loss')
    parser.add_argument('--exp', type=int, default=0, help='experiment id')
    opt = parser.parse_args()
    print(opt)

    os.makedirs('saved_models_' + str(opt.exp) + '/', exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    loss_geom = SamplesLoss("sinkhorn", p=2, blur=.01, scaling=.95, verbose=False)

    net = ResNet1D(num_classes=len(unique_labels))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = net.to(device)

    print_model_layers(net)

    Source_train_dset = PytorchDataSet(Source_train, len_features)
    Target_train_dset = PytorchDataSet(Target_train, len_features)
    Target_test_dset = PytorchDataSet(Target_test, len_features)

    train_loaderA = DataLoader(Source_train_dset, batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)
    train_loaderB = DataLoader(Target_train_dset, batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(Target_test_dset, batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    Lambda = opt.Lambda
    for epoch in range(opt.epoch, opt.n_epochs):
        net.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for (X1, Y1), (X2, Y2) in zip(train_loaderA, train_loaderB):
            X1, Y1, X2, Y2 = X1.to(device), Y1.to(device), X2.to(device), Y2.to(device)

            optimizer.zero_grad()

            outputs1 = net(X1)
            feat1 = nn.Sequential(*list(net.children())[:-1])(X1)
            feat2 = nn.Sequential(*list(net.children())[:-1])(X2)

            loss_g = loss_geom(feat1.detach().squeeze(), feat2.squeeze())
            loss_c = criterion(outputs1, Y1)
            loss_t = loss_c + loss_g * Lambda

            loss_t.backward()
            optimizer.step()

            running_loss += loss_t.item()
            pred_y = outputs1.cpu().detach().numpy()
            pred_y = np.argmax(pred_y, axis=1)
            total_correct += (pred_y == Y1.cpu().numpy()).sum()
            total_samples += Y1.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loaderA)
        epoch_acc = total_correct / float(total_samples)
        print(f"Epoch [{epoch}/{opt.n_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        torch.save(net.state_dict(), 'saved_models_' + str(opt.exp) + '/' + str(epoch) + '.pth')
        Eval(epoch, opt.exp, test_loader, unique_labels)
        


if __name__ == '__main__':
    main()