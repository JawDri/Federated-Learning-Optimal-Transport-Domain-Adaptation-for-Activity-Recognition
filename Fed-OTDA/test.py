import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch
from torchvision import models
from train_OT_Fed import ResNet1D
import copy
from sklearn.metrics import f1_score  # Added for F1-score calculation


def Eval(epoch, exp, test_loaderA, unique_labels, global_net): 
    os.makedirs('Eval_res' + str(exp) + '/', exist_ok=True)

    # models
    net = copy.deepcopy(global_net)
     
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = net.to(device)  
    
    net.eval()
    correct = 0
    total = 0
    all_preds = []  # Added to store all predictions for F1-score
    all_labels = []  # Added to store all true labels for F1-score
    
    with torch.no_grad():
        for X1, Y1 in test_loaderA:
            X1 = X1.to(device)             
            Y1 = Y1.to(device)             
            output = net(X1)     
            pred_y = output.cpu().detach().numpy()
            pred_y = np.argmax(pred_y, axis=1)
            correct += (pred_y == Y1.cpu().numpy()).sum()
            total += Y1.size(0)
            all_preds.extend(pred_y)  # Store predictions
            all_labels.extend(Y1.cpu().numpy())  # Store true labels
    
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Compute F1-score

    

    return correct / float(total), f1




