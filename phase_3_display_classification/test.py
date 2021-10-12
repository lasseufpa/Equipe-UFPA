import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    testset_folder = "C:\\Users\\Luan\\Documents\\GitHub\\phase3-display-digits\\digitsJ_test\\sign\\"

    image_transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])

    testset = datasets.ImageFolder(root=testset_folder, transform=image_transforms)
    test_loader = DataLoader(dataset=testset, shuffle=False, batch_size=1)

    model = torch.load("classifiers\\best-model-sign.pt")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            y_test_pred = torch.log_softmax(y_test_pred, dim=1)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(int(y_pred_tag.cpu().numpy()))
            y_true_list.append(int(y_batch.cpu().numpy()))
            
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list))#.rename(columns=range(10), index=range(10))
    fig, ax = plt.subplots(figsize=(7,5))         
    sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
    plt.show()

    y_true_list = np.array(y_true_list)
    y_pred_list = np.array(y_pred_list)

    print("Precision = {}%".format(np.sum(y_true_list==y_pred_list)/len(y_true_list)))