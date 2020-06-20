
# ## PyTorch-Tutorial 
# ## Chapter - 1 (The Classification)

# ## Introduction
# Pytorch tutorial is a series of tutorials created by me to explain the basic aspects of PyTorch and its implementation. PyTorch is complex to implement but not difficult. If you see it as a way of documentation or documenting a program, then things get much easier to understand. The most interesting part of this series is that I am also a beginner with PyTorch, so what's difficult me I expect to be difficult for five other individuals also, so at least I expect my tutorial could help five other individuals to implement PyTorch.<br>
# In this chapter, I have implemented an image classification problem with the help of PyTorch. Here I have explained everything in the most basic way possible so that you could also understand them easily.

# ## Index
# The things that are explained in this classification tutorial are given below.
# * Creating a custom dataset
# * Creating a neural network in PyTorch
# * Training neural network in PyTorch
# * Plotting of loss and accuracy curve
# * evaluation of performance

# ## Data-set
# Dataset used - [Arthropod Taxonomy Orders Object Detection Dataset](https://www.kaggle.com/mistag/arthropod-taxonomy-orders-object-detection-dataset)

# In[1]:

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# In[2]:

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[3]:


BASE_PATH = 'images/'


# In[4]:

#data = pd.read_csv('labels.csv') # Images, labels

images = []
labels = []

for file in os.listdir(BASE_PATH):
    
    file_class = file[0:file.find('_')]
    
    images.append(file)
    labels.append(file_class)

data = {'image': images, 'label': labels}
data = pd.DataFrame(data)

# In[5]:


lb = LabelEncoder()
data['encoded_label'] = lb.fit_transform(data['label'])

classes = pd.Series(data['label'].values, index = data['encoded_label']).to_dict()


# ## Spliting of Dataset

# In[6]:

batch_size = 128
validation_split = 0.3
shuffle_dataset = True
random_seed = 42

# Create data indices for training and validation splits
train, validation = train_test_split(data['label'], stratify = data['label'], test_size = validation_split)
train_indices = list(train.index)
validation_indices = list(validation.index)

# Create PyTorch data samplers
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)


# ## Transforms
# Transforms are common image transformations. They can be chained together using **Compose**.
# ## Normalization
# Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel of the input torch.*Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]
# Convert a PIL Image or numpy.ndarray to tensor.
# 
# ## ToTensor
# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8

# In[9]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


#  ## Create custom dataset class
#  A dataset must contain following functions to be used by data loader later on.
# 
# * __init__() function is where the initial logic happens like reading a csv, assigning transforms etc.
# * __getitem__() function returns the data and labels. This function is called from dataloader like this:
# 
# > img, label = MyCustomDataset.__getitem__(99)  # For 99th item
# 
# <br>
# An important thing to note is that __getitem__() return a specific type for a single data point (like a tensor, numpy array etc.), otherwise, in the data loader you will get an error like:
# 
# > TypeError: batch must contain tensors, numbers, dicts or lists; found 
# > class 'PIL.PngImagePlugin.PngImageFile'
# 
# <br>
# Credits: [PyTorch Custom Dataset Examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples)

# In[10]:


class LoadDataset(Dataset):
    
    def __init__(self, img_data, img_path, transform=None):
        
        self.img_path = img_path
        self.transform = transform
        self.img_data = img_data
    
    def __len__(self):
        
        return len(self.img_data)
    
    def __getitem__(self, index):
        
        img_name = os.path.join(self.img_path, self.img_data.loc[index, 'image'])
        
        image = Image.open(img_name)
        image = image.convert('RGB')
        image = image.resize((300, 300))
        label = torch.tensor(self.img_data.loc[index, 'encoded_label'])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# In[11]:


# Create PyTorch data loaders

dataset = LoadDataset(data, BASE_PATH, transform)

train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler = train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, sampler = validation_sampler)


# In[13]:


def img_display(img):
    img = img / 2 + 0.5 # Unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


# ## Visualisation
# Visualising the elements of the dataset

# In[14]:


# Display random training images
data_iter = iter(train_loader)
images, labels = data_iter.next()

fig, axis = plt.subplots(3, 5, figsize=(15, 10))

for i, ax in enumerate(axis.flat):
    
    with torch.no_grad():
        
        image, label = images[i], labels[i]
        ax.imshow(img_display(image))
        ax.set(title = f"{classes[label.item()]}")


# ## The Neural Network
# In the **Net** class created below, we have constructed a neural network.
# * Inside the **init()** method you declare each layer with a unique layer name.
# For every unique layer, declaring its input features and output features is a must.
# At least the input feature is a must for some of the layers like batch normalization.
# Inside the **forward(self, x)** method you need to connect the layers that were declared in the init method.
# One thing must be kept in mind that the output feature of one layer is an input feature of its next connecting layer.

# In[15]:


class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        
        # 3 input image channel, 16 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.4)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
    
    def forward(self, x):
        
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.dropout(self.batchnorm2(self.pool(x)))
        x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.conv4(x))
        x = x.view(-1, 64*5*5) # Flatten layer
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(self.fc3(x),dim = 1)
        
        return x


# In[16]:

if str(device) == 'cuda:0':
    model = Net().to(device)
else:
    model = Net()

print(model)


# ## CrossEntropyLoss
# It is useful when training a classification problem with C classes. If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes. This is particularly useful when you have an unbalanced training set. It is a prototpe of categorical crossentropy in keras.
# In case of **Binary classification** use **BCELoss(Binary Cross Entropy)** or BCEWithLogitsLoss.

# In[17]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)


# In[18]:


def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()


# ## Training the Network
# In the cell below, it is explained how to train your model with epochs.
# In the "train_loss" and "val_loss" the training loss and validation loss are stored respectively after every epoch.
# Similarly in case of training accuracy and validation accuracy also, the same thing happens.
# Just remember while validation the weights are not upgraded thats why we use **" with torch.no_grad() "**.
# Here, **"Torch.max(x, dim=1)"** works same as **"np.argmax(x, axis=1)"**.
# We use **".item()"** to get the value inside the tensor.
# **torch.save(model.state_dict(), 'model_classification_tutorial.pt')** is used to save the PyTorch weight in the given directory.

# In[19]:


n_epochs = 20
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_loader)

for epoch in range(1, n_epochs+1):
    
    running_loss = 0.0
    # scheduler.step(epoch)
    correct = 0
    total = 0
    
    print(f'Epoch {epoch}\n')
    
    for batch_idx, (data_, target_) in enumerate(train_loader):
        
        if str(device) == 'cuda:0':
            data_, target_ = data_.to(device), target_.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        
        if (batch_idx) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    
    print(f'\nTrain loss: {np.mean(train_loss):.4f}, Train acc: {(100 * correct / total):.4f}')
    
    batch_loss = 0
    total_t = 0
    correct_t = 0
    
    with torch.no_grad():
        
        model.eval()
        
        for data_t, target_t in (validation_loader):
            
            if str(device) == 'cuda:0':
                data_t, target_t = data_t.to(device), target_t.to(device)
            
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        
        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss/len(validation_loader))
        network_learned = batch_loss < valid_loss_min
        
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
        
        # Saving the best weight 
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'model_classification.pt')
            print('Detected network improvement, saving current model')
    
    model.train()


# ## Accuracy and loss Curve

# In[20]:


fig = plt.figure(figsize=(20,10))
plt.title("Train - Validation Loss")
plt.plot( train_loss, label='train')
plt.plot( val_loss, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(loc='best')


# In[21]:


fig = plt.figure(figsize=(20,10))
plt.title("Train - Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')


# In[22]:


# Importing trained Network with better loss of validation
model.load_state_dict(torch.load('model_classification.pt'))


# ## Evaluation
# Evaluating the model performance through visualization

# In[23]:


dataiter = iter(validation_loader)
images, labels = dataiter.next()

# Viewing data examples used for training
fig, axis = plt.subplots(3, 5, figsize=(15, 10))

with torch.no_grad():
    
    model.eval()
    
    for ax, image, label in zip(axis.flat, images, labels):
        ax.imshow(img_display(image)) # add image
        image_tensor = image.unsqueeze_(0)
        output_ = model(image_tensor)
        output_ = output_.argmax()
        k = output_.item()==label.item()
        ax.set_title(str(classes[label.item()])+":" +str(k)) # add label
