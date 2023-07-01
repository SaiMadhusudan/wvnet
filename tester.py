import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms as T
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np
from PIL import Image
import csv
import torch.nn.functional as F
import os
print("importing of modules is done")

testset_path = "./val.csv"
batch_size = 64
L_rate = 0.0005

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32768, 10240)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10240,1024)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1024, 128)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

def cosine_similarity(x, y):
    x_normalized = F.normalize(x, dim=1)
    y_normalized = F.normalize(y, dim=1)
    similarity = torch.sum(x_normalized * y_normalized, dim=1, keepdim=True)
    return similarity

def contrastive_loss(output1, output2, label, margin=1.0):
    similarity = cosine_similarity(output1, output2)
    loss = torch.mean((1 - label) * torch.pow(similarity, 2) +
                      label * torch.pow(torch.clamp(margin - similarity, min=0.0), 2))
    return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = SiameseNetwork().to(device)

import os 
if os.path.isfile('net.pth'):
    net.load_state_dict(torch.load('net.pth'))
    print("loaded model")

criterion = contrastive_loss
optimizer = optim.Adam(net.parameters(), lr=L_rate)


lines = []
first_line = True
with open(testset_path,'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if(first_line):
            first_line = False
            continue
        lines.append(row)
f.close()
class ValidationSetGenerator(Dataset):
    def __init__(self,image_paths,transform=None):
        super(ValidationSetGenerator,self).__init__()
        transform = T.Compose([
        T.Resize((155,600)), # Resize the image to a specific size
        #add gray_scale
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
        ])

        self.image_paths = image_paths
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):

        curr_line = self.image_paths[index]

        img1_path = "./val/" + curr_line[0]
        img2_path = "./val/" + curr_line[1]
        label = float(curr_line[2])
        label = torch.tensor(label)

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1 , img2 , label
    
print("loading of val ")

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            inputs1, inputs2, labels = data
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            outputs1, outputs2 = model(inputs1, inputs2)
            distance = cosine_similarity(outputs1, outputs2)
            predicted = torch.round(distance).squeeze()  # Predict 1 if distance is close, 0 otherwise
            labels = labels.squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

newvalSet = ValidationSetGenerator(lines)
newval_dl = torch.utils.data.DataLoader(newvalSet,batch_size=16,shuffle=True)

accuracy = evaluate(net,newval_dl)
print("Validation accuracy: {:.2f}%".format(accuracy * 100))

