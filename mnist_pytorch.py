import torch
import torch.optim as optim

from net import Net
from train import *
from load_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 15

train_data = load_trainset()
test_data = load_testset()

train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=8)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=8)

net = Net().to(device)
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,15],gamma=0.05)


for i in range(epochs):
    scheduler.step()
    train(i,net,train_loader,optimizer,device)
    test(net,test_loader,device)
