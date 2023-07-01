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

dataset_path = "./train"
number_e = 3
testset_path = "./val.csv"
L_rate = 0.0005
batch_size = 64

print("data loading started")
class DatasetGenerator(Dataset):
    def __init__(self,dir_path,transform=None):
        super(DatasetGenerator,self).__init__()
        transform = T.Compose([
        T.Resize((155,600)), # Resize the image to a specific size
        #add gray_scale
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
        ])

        self.dir_path = dir_path
        # need to get dataset from dir_path
        self.dataset = datasets.ImageFolder(dir_path,transform=transform)
        self.transform = transform
        self.group_examples()

    def group_examples(self):
        # we will get the indexes of the examples for each class
        self.group_examples = {}
        for i in range(len(self.dataset)):
            _,label = self.dataset[i]
            if label not in self.group_examples:
                self.group_examples[label] = []
            self.group_examples[label].append(i)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        selected_class = np.random.randint(0,len(self.group_examples)-1)
        random_index1 = np.random.randint(0,len(self.group_examples[selected_class])-1)
        index_1 = self.group_examples[selected_class][random_index1]

        #get first image
        img1, label1 = self.dataset[index_1]

        if index % 2 == 0:
            random_index2 = np.random.randint(0,len(self.group_examples[selected_class])-1)

            while random_index1 == random_index2:
                random_index2 = np.random.randint(0,len(self.group_examples[selected_class])-1)


            index_2 = self.group_examples[selected_class][random_index2]
            img2, label2 = self.dataset[index_2]

            target = torch.tensor([1.])

        else:
            other_selected_class = np.random.randint(0,len(self.group_examples)-1)

            while other_selected_class == selected_class:
                other_selected_class = np.random.randint(0,len(self.group_examples)-1)

            random_index2 = np.random.randint(0,len(self.group_examples[other_selected_class])-1)
            index_2 = self.group_examples[other_selected_class][random_index2]
            img2, label2 = self.dataset[index_2]

            target = torch.tensor([0.])

        return img1, img2, target
    
# Load dataset
# divde it into train and val
# create a dataloader for both train and val

datasets = DatasetGenerator(dataset_path)
train_size = int(0.8 * len(datasets))
val_size = len(datasets) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(datasets, [train_size, val_size])
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size, shuffle=True)

print("data loading done")

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

for epoch in range(number_e):
    running_loss = 0.0
    loss = 0

    for i ,data in enumerate(train_dl,0):
        inputs1, inputs2, labels = data
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs1, outputs2 = net(inputs1, inputs2)
        loss = criterion(outputs1, outputs2, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(loss)

print("Training Completed")

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

accuracy = evaluate(net,val_dl)
print("Validation accuracy: {:.2f}%".format(accuracy * 100))



 