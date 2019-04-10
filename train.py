import torch.nn.functional as F
from torch.autograd import Variable
def train(epoch,net,train_loader,optimizer,device):
    net.train()
    for batch_idx, (data,labels) in enumerate(train_loader):
        data, labels = Variable(data.to(device)), Variable(labels.to(device))
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output,labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('train Epoch:{}\t[{}/{}({:.0f})%]\tLoss:{:.6f}'.format(
                epoch,batch_idx * len(data),len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            )

def test(net,test_loader,device):
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data.to(device),volatile=True), Variable(target.to(device))
        output = net(data)
        test_loss += F.nll_loss(output,target, size_average=False).item()
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


