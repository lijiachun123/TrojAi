import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, base=32, dense=512, num_classes=43):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, base, 3, padding=1)
        self.conv2 = nn.Conv2d(base, base, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(base, base*2, 3, padding=1)
        self.conv4 = nn.Conv2d(base*2, base*2, 3)
        self.conv5 = nn.Conv2d(base*2, base*4, 3, padding=1)
        self.conv6 = nn.Conv2d(base*4, base*4, 3)

        self.fc1 = nn.Linear(32 * 4 * 2 * 2, dense)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dense, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print("after cv1", x.shape)
        x = F.relu(self.conv2(x))
        #print("after cv2: ", x.shape)
        x = self.dropout(self.pool(x))
        #print("after pool1: ", x.shape)
        x = F.relu(self.conv4(F.relu(self.conv3(x))))
        #print("after cv4: ", x.shape)
        x = self.dropout(self.pool(x))
        #print("after pool2: ", x.shape)
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        #print("after cv 6", x.shape)
        x = self.dropout(self.pool(x))
        #print("before fc: ", x.shape)
        x = x.view(-1, 32 * 4 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        tgt_layer = x
        x = self.fc2(x)
        return x, tgt_layer


def c6f2():
    return CNN()
