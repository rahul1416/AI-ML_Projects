import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

def load_data():
    train_data = datasets.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )
    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
        
        'test'  : torch.utils.data.DataLoader(test_data, 
                                            batch_size=100, 
                                            shuffle=True, 
                                            num_workers=1),
    }
    return loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaders = load_data()
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
model = CNN().to(device)
model.load_state_dict(torch.load("model/model.pth"))
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
n_epochs = 1
def train(model,loss_func,optimizer,n_epochs):
    model.train()
    for epoch in range(n_epochs) :
        for batch_idx, (data, target) in enumerate(loaders['train']):
            data = data.to(device)
            target = target.to(device)
            y_pred = model(data)
            loss = loss_func(y_pred, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Training Loss in epoch {epoch+ 1} is {loss}")
    return torch.save(model.state_dict(), "model/model.pth")
train(model,loss_func,optimizer,n_epochs)