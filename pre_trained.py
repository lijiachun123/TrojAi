from __future__ import division, print_function

import argparse, os
import torch
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import sys
sys.path.append("..")
from model import ResNet18
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

#from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Average(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count

    def __str__(self):
        return '{:.6f}'.format(self.average)


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        predictions = output.data.argmax(dim=1)
        correct = predictions.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)


class Trainer(object):
    def __init__(self, net, optimizer, train_loader, test_loader, device, scheduler):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.scheduler = scheduler

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            self.scheduler.step()
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()
            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc))


    def train(self):
        train_loss = Average()
        train_acc = Accuracy()
        print("_______")
        self.net.train()
        a=0
        for data, label in self.train_loader:
            data = data.to(self.device)
            label = label.to(self.device)
            a+=1
            output, _ = self.net(data)
            loss = self.loss_func(output, label)
            #print(a, loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, label)
            #print(a, train_acc)
        return train_loss, train_acc

    def evaluate(self):
        test_loss = Average()
        test_acc = Accuracy()
        
        self.net.eval()

        with torch.no_grad():
            for data, label in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                output, _ = self.net(data)
                loss = self.loss_func(output, label)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, label)

        return test_loss, test_acc


def get_dataloader(root, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = ImageFolder(root=root, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = ImageFolder(root='dataset/test', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def run(args):
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    net = ResNet18().to(device)

    # optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9,weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=[0.9, 0.99], weight_decay=1*10e-5)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8)
    train_loader, test_loader = get_dataloader(args.root, args.batch_size)
    trainer = Trainer(net, optimizer, train_loader, test_loader, device, scheduler)
    trainer.fit(args.epochs)
    torch.save(net,"baseline_gt_resnet_ep10.pth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001)
    parser.add_argument('--root', type=str, default='dataset/train')
    parser.add_argument('--batch-size', type=int, default=64)


    args = parser.parse_args()
    print(args)
    run(args)
    # test(args)

if __name__ == '__main__':
    main()

