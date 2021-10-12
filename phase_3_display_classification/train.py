import torch
import glob, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy as np

from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

from model import Net

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc) * 100
    return acc

if __name__ == "__main__":
    trainset_folder = "C:\\Users\\Luan\\Documents\\GitHub\\phase3-display-digits\\digitsJ_train\\sign\\"

    if os.path.isdir("./classifier/"):
        pass
    else:
        os.mkdir("./classifier/")

    image_transforms = transforms.Compose([
            # transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])

    trainset = datasets.ImageFolder(root=trainset_folder, transform=image_transforms)

    dataset_size = len(trainset)
    dataset_indices = list(range(dataset_size))

    np.random.shuffle(dataset_indices)

    val_split_index = int(np.floor(0.2 * dataset_size))

    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset=trainset, shuffle=False, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(dataset=trainset, shuffle=False, batch_size=8, sampler=val_sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(n_out=4)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    best_accuracy = 0
    best_loss = 1000

    print("Begin training.")
    for e in range(1, 11):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch).squeeze()
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        # VALIDATION
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch).squeeze()
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += train_loss.item()
                val_epoch_acc += train_acc.item()
        accuracy = val_epoch_acc/len(val_loader)
        loss = val_epoch_loss/len(val_loader)
        if accuracy > best_accuracy: 
            torch.save(model, './classifier/best-model.pt')
            best_accuracy = accuracy
        elif loss < best_loss:
            torch.save(model, './classifier/best-model.pt')
            best_loss = loss
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')