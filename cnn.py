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
    print(f"Using device: {self.device}")  # Confirm if using GPU or CPU
      # feel free to adjust the convolutional layers, learning rate, batch sizes, epochs
      # convolutional layer (32 filters of 3x3) with BathNorm and ReLU
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
    # Dropout helps reduce overfitting 
    self.dropout = nn.Dropout(0.3)
    
    input_dims = self.calc_input_dims()
      
    # fed into a linear layer set (dealing with 10 digits)
    self.fc1 = nn.Linear(input_dims, self.num_classes)
    # Adam optimizer (glorified gradient descent) automatically adjusts the learning rate in 
    # terms of momentum 
    # Changed optimizer to AdamW for better regularization (adds weight decay regularization)
    self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)  # LR scheduler

    self.loss = nn.CrossEntropyLoss()
      # mean squared error (goal is to minimize this)
    self.to(self.device)
    self.get_data()

  def calc_input_dims(self):
    with torch.no_grad():
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
    batch_data = batch_data.to(self.device)

    batch_data = F.relu(self.bn1(self.conv1(batch_data)))
    batch_data = F.relu(self.bn2(self.conv2(batch_data)))
    batch_data = self.dropout(F.relu(self.bn3(self.conv3(batch_data))))
    batch_data = self.maxpool1(batch_data)

    batch_data = F.relu(self.bn4(self.conv4(batch_data)))
    batch_data = F.relu(self.bn5(self.conv5(batch_data)))
    batch_data = self.dropout(F.relu(self.bn6(self.conv6(batch_data))))
    batch_data = self.maxpool2(batch_data)

    batch_data = batch_data.view(batch_data.size(0), -1)
    output = self.fc1(batch_data)
    return output
    # dropout layers added after the 3rd and 6th convolution blocks to help reduce overfitting
    # by randomly zeroing out some neuron activations during training 

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
      for input, label in self.train_data_loader:
          self.optimizer.zero_grad()
          label = label.to(self.device)
          prediction = self.forward(input)
          loss = self.loss(prediction, label)

          pred_labels = torch.argmax(F.softmax(prediction, dim=1), dim=1)
          acc = (pred_labels == label).float().mean()
          ep_acc.append(acc.item())

          ep_loss += loss.item()
          loss.backward()
          self.optimizer.step()

          self.scheduler.step()  # Decreases the learning rate every few epochs to help model converge better 
          avg_loss = ep_loss / len(self.train_data_loader)
          avg_acc = np.mean(ep_acc)
          self.loss_history.append(avg_loss)
          self.acc_history.append(avg_acc)
          print(f"Epoch {i+1}: Loss = {avg_loss:.3f}, Accuracy = {avg_acc:.3f}")

  def _test(self):
    self.eval()
    ep_loss = 0
    ep_acc = []
    with torch.no_grad(): #Prevents gradient calculations during testing and improves speed/reduces memory usage
      for input, label in self.test_data_loader:
        label = label.to(self.device)
        prediction = self.forward(input)
        loss = self.loss(prediction, label)

        pred_labels = torch.argmax(F.softmax(prediction, dim=1), dim=1)
        acc = (pred_labels == label).float().mean()
        ep_acc.append(acc.item())

        ep_loss += loss.item()

      avg_loss = ep_loss / len(self.test_data_loader)
      avg_acc = np.mean(ep_acc)
      print(f"Test Results: Loss = {avg_loss:.3f}, Accuracy = {avg_acc:.3f}")


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

  #plt.show()
  print("It took {} seconds to train the network.".format(end-start))
  network._test()
