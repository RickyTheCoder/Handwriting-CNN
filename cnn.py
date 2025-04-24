import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms
# import ToTensor
import numpy as np
import matplotlib.pyplot as plt

import time

class TransposeTransform:
  def __call__(self, img):
      # transpose function from pytorch
    return torch.transpose(img, 1,2)
    # working with black and white input (either 0 or 1)
class BlackWhiteTransform:
  #this will turn the tensor to black and white (no grayscale)
  def __call__(self, img):
      # 0.59 and below is white anything above is black 
    return (img >= 0.6).float()

# Define the transformation to convert images into tensors
transform = transforms.Compose([
    # resize to be 28 by 28 
  transforms.Resize((28, 28)), 
    #turn input data into a tensor (perform calculations with pytorch)
  transforms.ToTensor(),
  BlackWhiteTransform(),
    # digits are sideways so we can put them back to normal
  TransposeTransform()
  ])


class CNN(nn.Module):
    # depending on the split of the EMNIST database, set 'num_classes' to expected outputs
    # since we're doing digits, No.Classes is 10
  def __init__(self, lr=0.001, batch_size=64, num_classes=62):
    super(CNN, self).__init__()
    self.lr = lr
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.loss_history = []
    self.acc_history = []
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      # feel free to adjust the convolutional layers, learning rate, batch sizes, epochs
      # convolutional layer (32 filters of 3x3)
    self.conv1 = nn.Conv2d(1, 32, 3)
      # batch normalization 
    self.bn1 = nn.BatchNorm2d(32)
      # another conv layer 
    self.conv2 = nn.Conv2d(32, 32, 3)
      # more batch normalization 
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, 3)
    self.bn3 = nn.BatchNorm2d(32)
      # max pooling 
    self.maxpool1 = nn.MaxPool2d(2)
    self.conv4 = nn.Conv2d(32, 64, 3)
    self.bn4 = nn.BatchNorm2d(64)
    self.conv5 = nn.Conv2d(64, 64, 3)
    self.bn5 = nn.BatchNorm2d(64)
    self.conv6 = nn.Conv2d(64, 64, 3)
    self.bn6 = nn.BatchNorm2d(64)
    self.maxpool2 = nn.MaxPool2d(2)

    input_dims = self.calc_input_dims()
      
    # fed into a linear layer set (dealing with 10 digits)
    self.fc1 = nn.Linear(input_dims, self.num_classes)
    # Adam optimizer (glorified gradient descent) automatically adjusts the learning rate in 
    # terms of momentum 
    self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    self.loss = nn.CrossEntropyLoss()
      # mean squared error (goal is to minimize this)
    self.to(self.device)
    self.get_data()

  def calc_input_dims(self):
    batch_data = torch.zeros((1, 1, 28, 28))
    batch_data = self.conv1(batch_data)
    #batch_data = self.bn1(batch_data)
    batch_data = self.conv2(batch_data)
    #batch_data = self.bn2(batch_data)
    batch_data = self.conv3(batch_data)

    batch_data = self.maxpool1(batch_data)
    batch_data = self.conv4(batch_data)
    batch_data = self.conv5(batch_data)
    batch_data = self.conv6(batch_data)
    batch_data = self.maxpool2(batch_data)

    return int(np.prod(batch_data.size()))

  def forward(self, batch_data):
    batch_data = torch.tensor(batch_data).to(self.device)

    batch_data = self.conv1(batch_data)
    batch_data = self.bn1(batch_data)
    batch_data = F.relu(batch_data)

    batch_data = self.conv2(batch_data)
    batch_data = self.bn2(batch_data)
    batch_data = F.relu(batch_data)

    batch_data = self.conv3(batch_data)
    batch_data = self.bn3(batch_data)
    batch_data = F.relu(batch_data)

    batch_data = self.maxpool1(batch_data)

    batch_data = self.conv4(batch_data)
    batch_data = self.bn4(batch_data)
    batch_data = F.relu(batch_data)

    batch_data = self.conv5(batch_data)
    batch_data = self.bn5(batch_data)
    batch_data = F.relu(batch_data)

    batch_data = self.conv6(batch_data)
    batch_data = self.bn6(batch_data)
    batch_data = F.relu(batch_data)

    batch_data = self.maxpool2(batch_data)

    batch_data = batch_data.view(batch_data.size()[0], -1)

    classes = self.fc1(batch_data)

    return classes

# Changed get_data function to train on EMNIST dataset
    
  def get_data(self):
    SplitType = "byclass"
    train_data = EMNIST('..//emnist', split=SplitType, train=True,
                             download=True, transform=transform)
      # transform function (when data loaded in, its in its own form
      # we want to transform the form we want 
                             
    self.train_data_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=8)   
    test_data = EMNIST('..//emnist', split=SplitType, train=False,
                             download=True, transform=transform)
                             

    self.test_data_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=8)  

  def _train(self, epochs):
    self.train()
    for i in range(epochs):
      ep_loss = 0
      ep_acc = []
      for j, (input, label) in enumerate(self.train_data_loader):
        self.optimizer.zero_grad()
        label = label.to(self.device)
        prediction = self.forward(input)
        loss = self.loss(prediction, label)
        prediction = F.softmax(prediction, dim=1)
        classes = torch.argmax(prediction, dim=1)
        wrong = torch.where(classes != label,
                        torch.tensor([1.]).to(self.device),
                        torch.tensor([0.]).to(self.device))
        acc = 1 - torch.sum(wrong) / self.batch_size

        ep_acc.append(acc.item())
        self.acc_history.append(acc.item())
        ep_loss += loss.item()
        loss.backward()
        self.optimizer.step()
      print('Finish epoch', i, 'total loss %.3f' % ep_loss,
              'accuracy %.3f' % np.mean(ep_acc))
      self.loss_history.append(ep_loss)

  def _test(self):
    self.eval()

    ep_loss = 0
    ep_acc = []
    for j, (input, label) in enumerate(self.test_data_loader):
      label = label.to(self.device)
      prediction = self.forward(input)
      loss = self.loss(prediction, label)
      prediction = F.softmax(prediction, dim=1)
      classes = torch.argmax(prediction, dim=1)
      wrong = torch.where(classes != label,
                      torch.tensor([1.]).to(self.device),
                      torch.tensor([0.]).to(self.device))
      acc = 1 - torch.sum(wrong) / self.batch_size

      ep_acc.append(acc.item())

      ep_loss += loss.item()

    print('total loss %.3f' % ep_loss,
                'accuracy %.3f' % np.mean(ep_acc))


if __name__ == '__main__':
  start = time.time()
    # changed epochs to 'num_classes'
  network = CNN(lr=0.001, batch_size=64)
  network._train(epochs=20)
  end = time.time()
  #torch.save(network.state_dict(),"CNN-weights.pt")
  #plt.plot(network.loss_history)
  #plt.show()
  #plt.plot(network.acc_history)
  #plt.show()
  print("It took {} seconds to train the network.".format(end-start))
  network._test()
