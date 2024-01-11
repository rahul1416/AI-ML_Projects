import pandas as pd 
import numpy as np 
import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader


df1 = pd.read_csv('data/digit-recognizer/train.csv')  
file_path = 'data/digit-recognizer/updated.csv'  # feedback data is in there where we stre the feedback of our inputs
if os.path.exists(file_path):
    df2 = pd.read_csv('data/digit-recognizer/updated.csv')
    df = pd.concat([df1, df2], axis=0)
    os.remove(file_path)
else:
    df = df1

df.to_csv('data/digit-recognizer/train.csv', index=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_test_split(df, test_size):
    test_size = int(len(df) * test_size)
    train_data, test_data = df[test_size:], df[:test_size]
    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    return train_data, test_data
def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std
def getData(data, batch_size, test_size):
    train, test = train_test_split(data, test_size)
    train_x = normalize(torch.from_numpy(train.iloc[:,1:].values.reshape(-1,1,28,28)).float())
    train_y = torch.from_numpy(train.iloc[:,0].values.reshape(-1,1)).squeeze(1).long()
    test_x = normalize(torch.from_numpy(test.iloc[:,1:].values.reshape(-1,1,28,28)).float())
    test_y = torch.from_numpy(test.iloc[:,0].values.reshape(-1,1)).squeeze(1).long()
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.Conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.Conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(500, 100)  # Calculate input size for fully connected layer
        self.fc2 = nn.Linear(100, 75)  # Additional layer
        self.fc3 = nn.Linear(75, 10)   # New output layer
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.Conv1(x), 2))
        x = F.relu(F.max_pool2d(self.Conv2_drop(self.Conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Pass through additional layer
        x = F.log_softmax(self.fc3(x), dim=1)  # New output layer with log_softmax
        return x

model = CNN()
model.load_state_dict(torch.load("model/model.pth"))
model.eval()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
n_epochs = 10

def train_model(model, train_loader, optimizer, loss_func, n_epochs, device):
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_func(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], '
                    f'Loss: {running_loss / (batch_idx + 1):.4f}, Accuracy: {100. * correct / total:.2f}%')
    torch.save(model.state_dict(), "model/model.pth")

train_loader, test_loader = getData(df, 32, 0.2)
train_model(model, train_loader, optimizer, loss_func, n_epochs, device)
