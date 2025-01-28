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
import copy
from sklearn.metrics import f1_score  # Added for F1-score calculation


# Load datasets
Source_train_1 = pd.read_csv("/content/drive/MyDrive/OTRUDA/data/Source_train_1.csv")
Source_train_2 = pd.read_csv("/content/drive/MyDrive/OTRUDA/data/Source_train_2.csv")
Source_train_3 = pd.read_csv("/content/drive/MyDrive/OTRUDA/data/Source_train_3.csv")


Target_train = pd.read_csv("/content/drive/MyDrive/OTRUDA/data/Target_train.csv")
Target_test = pd.read_csv("/content/drive/MyDrive/OTRUDA/data/Target_test.csv")


FEATURES_dset = [i for i in Source_train_1.columns if i != 'labels']
len_features = len(FEATURES_dset)
unique_labels = Source_train_1['labels'].unique().tolist()
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
        self.layer2 = self._make_custom_layer(64, 128, 1, stride=1)
        self.layer3 = self._make_custom_layer(128, 256, 1, stride=1)
        self.layer4 = self._make_custom_layer(256, 512, 1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16384, num_classes)

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
        #x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Define federated learning functions
def local_train(net, global_net, train_loaderA, train_loaderB, device, optimizer, scheduler, criterion, loss_geom, Lambda, mu):
    net.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []  # Added to store all predictions for F1-score
    all_labels = []  # Added to store all true labels for F1-score

    for (X1, Y1), (X2, Y2) in zip(train_loaderA, train_loaderB):
        X1, Y1, X2, Y2 = X1.to(device), Y1.to(device), X2.to(device), Y2.to(device)

        optimizer.zero_grad()

        outputs1 = net(X1)
        feat1 = nn.Sequential(*list(net.children())[:-1])(X1)
        feat2 = nn.Sequential(*list(net.children())[:-1])(X2)

        loss_g = loss_geom(feat1.squeeze(), feat2.squeeze())
        loss_c = criterion(outputs1, Y1)
        loss_t = loss_c + loss_g * Lambda

        # FedProx proximal term
        prox_term = 0.0
        for param, global_param in zip(net.parameters(), global_net.parameters()):
            prox_term += ((param - global_param.to(device)) ** 2).sum()
        loss_t += (mu / 2) * prox_term

        loss_t.backward()
        optimizer.step()

        running_loss += loss_t.item()
        pred_y = outputs1.cpu().detach().numpy()
        pred_y = np.argmax(pred_y, axis=1)
        total_correct += (pred_y == Y1.cpu().numpy()).sum()
        total_samples += Y1.size(0)
    
        all_preds.extend(pred_y)  # Store predictions
        all_labels.extend(Y1.cpu().numpy())  # Store true labels

    scheduler.step()

    epoch_loss = running_loss / len(train_loaderA)
    epoch_acc = total_correct / float(total_samples)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')  # Compute F1-score
    
    return epoch_loss, epoch_acc, epoch_f1


def federated_averaging(global_model, client_models):
    global_state_dict = global_model.state_dict()
    client_weights = [1/len(client_models) for i in client_models]
    
    for key in global_state_dict.keys():
        weighted_sum = sum(client_weights[i] * client_models[i].state_dict()[key].float() 
                           for i in range(len(client_models)))
        global_state_dict[key] = weighted_sum
    
    global_model.load_state_dict(global_state_dict)
    return global_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--Lambda', type=float, default=0.1, help='lambda value for OT loss')
    parser.add_argument('--mu', type=float, default=0.0001, help='proximal term coefficient for FedProx')
    parser.add_argument('--exp', type=int, default=0, help='experiment id')
    opt = parser.parse_args()
    print(opt)

    os.makedirs('saved_models_' + str(opt.exp) + '/', exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    loss_geom = SamplesLoss("sinkhorn", p=2, blur=.01, scaling=.95, verbose=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Federated learning setup
    client_loaders = []

    Source_train_dset_1 = PytorchDataSet(Source_train_1, len_features)
    Target_train_dset = PytorchDataSet(Target_train, len_features)
    train_loader_1 = DataLoader(Source_train_dset_1, batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)
    train_loaderB = DataLoader(Target_train_dset, batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)
    client_loaders.append((train_loader_1, train_loaderB))


    Source_train_dset_2 = PytorchDataSet(Source_train_2, len_features)
    train_loader_2 = DataLoader(Source_train_dset_2, batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)
    client_loaders.append((train_loader_2, train_loaderB))


    Source_train_dset_3 = PytorchDataSet(Source_train_3, len_features)
    train_loader_3 = DataLoader(Source_train_dset_3, batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)
    client_loaders.append((train_loader_3, train_loaderB))

    test_loader = DataLoader(PytorchDataSet(Target_test, len_features), batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)

    global_net = ResNet1D(num_classes=len(unique_labels)).to(device)
    print(global_net)
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        global_net = nn.DataParallel(global_net).to(device)

    for epoch in range(opt.epoch, opt.n_epochs):
        client_models = []
        for i, (train_loaderA, train_loaderB) in enumerate(client_loaders):
            # Initialize the client model and optimizer
            local_net = copy.deepcopy(global_net)
            optimizer = torch.optim.SGD(local_net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            
            # Local training on the client with FedProx
            epoch_loss, epoch_acc, epoch_fscore = local_train(local_net, global_net, train_loaderA, train_loaderB, device, optimizer, scheduler, criterion, loss_geom, opt.Lambda, opt.mu)
            print(f"Client {i} Epoch [{epoch}/{opt.n_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print(f"Client {i} Epoch [{epoch}/{opt.n_epochs}] Loss: {epoch_loss:.4f} F1 score: {epoch_fscore:.4f}")
            
            # Append the trained client model for federated averaging
            client_models.append(copy.deepcopy(local_net))
        
        # Federated averaging to update the global model
        global_net = federated_averaging(copy.deepcopy(global_net), client_models)

        # Test the global model
        accuracy, f1 = Eval(epoch, opt.exp, test_loader, unique_labels, copy.deepcopy(global_net),)

        print(f"Global model with accuracy: {accuracy:.4f}")
        print(f"Global model with f1 score: {f1:.4f}")


if __name__ == '__main__':
    main()
