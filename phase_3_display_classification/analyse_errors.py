import torch
import torchvision
import cv2
import numpy as np
import os
import sys

from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyse_errors.py <0, 1, 2 or 3>")
        exit(0)
    else:
        net_option = int(sys.argv[1])
    nets = {0:"digits_a", 1:"digits_b", 2:"digits_d", 3:"sign"}
    models = {0:"digits-a", 1:"digits-b", 2:"digits-d", 3:"sign"}
    dataset_folder = "C:\\Users\\Luan\\Documents\\GitHub\\phase3-display-digits\\digitsJ_train\\{}\\".format(nets[net_option])

    image_transforms =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])

    dataset = datasets.ImageFolder(root=dataset_folder, transform=image_transforms)
    test_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)

    model = torch.load("classifiers\\best-model-{}.pt".format(models[net_option]))
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
            
    y_true_list = np.array(y_true_list)
    y_pred_list = np.array(y_pred_list)

    idx = (y_true_list != y_pred_list)
    files = np.array(dataset.imgs)[idx]

    print("Erros/Images: {}/{}".format(len(files),len(y_true_list)))
    
    for i in range(len(files)):
        img = cv2.imread("{}".format(files[i][0]))
        cv2.imshow("Error: {} - Label: {}".format(i,files[i][1]), img)
        k=cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == ord('p'):
            cv2.destroyAllWindows()
        elif k == ord('d'):
            os.remove(files[i][0])
            cv2.destroyAllWindows()
    