import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import sklearn
from sklearn.model_selection import train_test_split

import PIL 
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
from torchinfo import summary 

import torch.optim as optim
from IPython.display import Image
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import cv2
import pickle

def get_image_paths_and_labels(clean_dir, mal_dir):
    clean_images = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    mal_images = [os.path.join(mal_dir, f) for f in os.listdir(mal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    images = clean_images + mal_images
    labels=np.array([0]*len(clean_images)+[1]*len(mal_images))

    return images, labels

#Chargement des images

clean = '/root/Data/Benign/Graphes_entropie'
mal = '/root/Data/Malicious/Graphes_entropie'

images, labels = get_image_paths_and_labels(clean, mal)

#Les classes sont des Malwares ou des Cleanwares :
classes = ['0', '1']

images_tv, images_test, y_tv, y_test  = train_test_split(images, labels, shuffle=True, test_size=0.2, random_state=123)
images_train, images_val, y_train, y_val  = train_test_split(images_tv, y_tv, shuffle=True, test_size=0.25, random_state=123)

class CT_Dataset(Dataset):
    def __init__(self, img_path, img_labels, img_transforms=None, grayscale=True):
        self.img_path = img_path
        self.img_labels = torch.Tensor(img_labels)
        
        if img_transforms is None:
            if grayscale:
                self.transforms = transforms.Compose([
                    transforms.Grayscale(), 
                    transforms.Resize((250,250)),
                    transforms.ToTensor()
                ])
        
            else :
                self.transforms = transforms.Compose([
                    transforms.Resize((250,250)),
                    transforms.ToTensor()
                ])
        
        else:
            self.transforms = img_transforms

    
    def __getitem__(self, index):
        #load image
        cur_path=self.img_path[index]
        cur_img=PIL.Image.open(cur_path).convert('RGB')
        cur_img=self.transforms(cur_img)

        return cur_img, self.img_labels[index]

    def __len__(self):
        return len(self.img_path)

# define CNN mode
class Convnet(nn.Module):
    def __init__(self, dropout=0.5):
        super(Convnet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=12800, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = self.classifier(x)
        return x

#define training function

def train_model(model, train_dataset, val_dataset, test_dataset, device, lr=0.0001, epochs=30, batch_size=32, l2=0.00001, gamma=0.5, patience=7):
    
    model = model.to(device)

    #construct dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #history
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    #set up loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=patience, gamma=gamma)

    #training loop
    print("Training Start:")
    for epoch in range(epochs):
        model.train()
        
        train_loss=0
        train_acc=0
        val_loss=0
        val_acc=0

        for i, (images, labels) in enumerate(train_loader):
            #reshape images
            images=images.to(device)
            labels=labels.to(device)
            #forward
            outputs=model(images).view(-1)
            pred=torch.sigmoid(outputs)
            pred=torch.round(pred)

            cur_train_loss=criterion(outputs, labels)
            cur_train_acc=(pred==labels).sum().item()/batch_size

            #backward
            cur_train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #loss
            train_loss += cur_train_loss
            train_acc += cur_train_acc

        #valid
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images=images.to(device)
                labels=labels.to(device)
                outputs=model(images).view(-1)
                cur_valid_loss=criterion(outputs, labels)
                val_loss += cur_valid_loss
                pred=torch.sigmoid(outputs)
                pred=torch.round(pred)
                val_acc += (pred==labels).sum().item()/batch_size
                    
        scheduler.step()

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
                    
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)

        print(f"Epoch:{epoch + 1} / {epochs}, lr: {optimizer.param_groups[0]['lr']:.5f} train loss:{train_loss:.5f}, train acc: {train_acc:.5f}, valid loss:{val_loss:.5f}, valid acc:{val_acc:.5f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    test_acc=0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
                        
            outputs=model(images)

            pred=torch.sigmoid(outputs)
            pred=torch.round(pred)
            test_acc += (pred == labels).sum().item()

    print(f'Test Accuracy : {(test_acc/len(test_loader))}')

    return history

train_dataset=CT_Dataset(img_path=images_train, img_labels=y_train)
val_dataset=CT_Dataset(img_path=images_val, img_labels=y_val)
test_dataset=CT_Dataset(img_path=images_test, img_labels=y_test)

device = torch.device("cpu")

cnn_model = Convnet(dropout=0.5)
hist= train_model(cnn_model, train_dataset, val_dataset, test_dataset, device, lr=0.0002, batch_size=32, epochs=5, l2=0.09, patience=5)

# Save the trained model to a file
with open('cnn_model_entropie.pkl', 'wb') as f:
    pickle.dump(cnn_model, f)





