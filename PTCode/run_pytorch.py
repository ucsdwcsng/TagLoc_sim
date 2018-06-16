
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import torchvision
from collections import defaultdict
import h5py
import scipy.io
from torch.autograd import Variable
import torch.optim as optim
import argparse
import os
from read_config import *


# In[ ]:


#Parser
parser = argparse.ArgumentParser(description='Run CNN Pipeline on config')
parser.add_argument('-c', '--config_file', help='Config file', required=True)
args = vars(parser.parse_args())

config=get_config_from_file(args['config_file'])
#Make directory
result_dir=config['result_dir']
if not os.path.exists(result_dir[0]):
    os.makedirs(result_dir[0], exist_ok=True)
channels = config['channels']


# In[ ]:


def load_data(filename,sigma='1'):
    arrays = {}
    f = h5py.File(filename,'r')
    # idx = [5,6,9,10,13,14]
    # idx = [0,1,2,4,5,6,8,9,10,12,13,14]
    for k, v in f.items():
        if(k=='features'):
            features = np.transpose(np.array(v))
            shape = features.shape
            if(shape[1]==16  and channels[0]=='rel'):
                idx = [0,1,2,4,5,6,8,9,10,12,13,14]
                features = torch.from_numpy(features[:,idx,:,:]).float()
            elif(shape[1]==16  and channels[0]=='abs'):
                idx = [3,7,11,15]
                features = torch.from_numpy(features[:,idx,:,:]).float()
            elif(shape[1]==16  and channels[0]=='both'):
                features = torch.from_numpy(features).float()
            elif(shape[1]==12 and channels[0]=='rel'):
                features = torch.from_numpy(features).float()
            elif(shape[1]==3 and channels[0]=='abs'):
                features = torch.from_numpy(features).float()
            else:
                raise ValueError('Something is wrong with your no of channels!')
        elif(k=='labels_gaussian_2d_1' and sigma=='1'):
            labels = torch.from_numpy(np.transpose(np.array(v))).float()
        elif(k=='labels_gaussian_2d_2' and sigma=='0.25'):
            labels = torch.from_numpy(np.transpose(np.array(v))).float()
        elif(k=='labels_gaussian_2d_5' and sigma=='0.5'):
            labels = torch.from_numpy(np.transpose(np.array(v))).float()
        elif(k=='labels_gaussian_2d'):
            labels = torch.from_numpy(np.transpose(np.array(v))).float()
    return features, labels


# In[ ]:


trainpath=config['train_filenames']
testpath=config['test_filenames']
sigma = config['sigma']
#trainpath = ['../ReflectionModel/datasets/crctd_dataset1_27_4_1.mat', '../ReflectionModel/datasets/crctd_dataset1_27_4_2.mat', '../ReflectionModel/datasets/crctd_dataset1_27_4_3.mat','../ReflectionModel/datasets/crctd_dataset1_27_4_4.mat','../ReflectionModel/datasets/crctd_dataset1_27_4_5.mat','../ReflectionModel/datasets/crctd_dataset1_27_4_6.mat']
#testpath = ['../ReflectionModel/datasets/dataset1_real27Feb.mat']

features_train,labels_train = load_data(trainpath[0],sigma[0])
for i in range(len(trainpath)-1):
    f,l = load_data(trainpath[i+1],sigma[0])
    features_train = torch.cat((features_train, f), 0)
    labels_train = torch.cat((labels_train, l), 0)
    
features_test,labels_test = load_data(testpath[0],sigma[0])
for i in range(len(testpath)-1):
    f,l = load_data(testpath[i+1],sigma[0])
    features_test = torch.cat((features_test, f), 0)
    labels_test = torch.cat((labels_test, l), 0)

# test_shape = labels_test.shape
# test_batch = test_shape[0]
    
train = torch.utils.data.TensorDataset(features_train, labels_train)
train_loader =torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

test = torch.utils.data.TensorDataset(features_test, labels_test)
test_loader =torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

print(features_train.shape)
print(labels_train.shape)
print(features_test.shape)
print(labels_test.shape)


# In[ ]:


# Define all the required constants here
_,n_channel,image_height,image_width = features_train.shape
depth = 12
depth_step = 2
eta = 0.001
n_epochs = 200
res_ep = 100


# In[ ]:


def train_test(cnn,eta,n_epochs,res_ep=100):
    
    optimiser = optim.Adam(cnn.parameters(),lr=eta)
    loss = torch.nn.MSELoss()
    for epoch in range(n_epochs):

        l_tr= 0.0
        loss_train = 0.0
        count = 0
        for i,data in enumerate(train_loader,0):
            count+=1
            inputs,labels = data
            inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
#             inputs,labels = Variable(inputs),Variable(labels)
            optimiser.zero_grad()

            preds = cnn(inputs)
            loss_tr = loss(preds,labels)
            loss_tr.backward()
            optimiser.step()

            l_tr += loss_tr.data[0]
            loss_train += loss_tr.data[0]
        print('%d Epoch Training loss: %f' %(epoch + 1, loss_train/count))
        label_out = np.array([])
        predict_out = np.array([])
        if epoch % res_ep ==0:
            loss_test = 0.0
            count = 0
            for i,data in enumerate(test_loader,0):
                images, labels = data
                images,labels = Variable(images.cuda()),Variable(labels.cuda())
                outputs = cnn(images)
                loss_te = loss(outputs,labels)
                loss_test += loss_te.data[0]
                if i==0:
                    label_out = np.squeeze(np.array(labels.data))
                    predict_out = np.squeeze(np.array(outputs.data))
                else:
                    label_out = np.append(label_out,np.squeeze(np.array(labels.data)),axis=0)
                    predict_out = np.append(predict_out,np.squeeze(np.array(outputs.data)),axis=0)
                count += 1
            scipy.io.savemat(result_dir[0]+'/test_real_27net1_'+str(epoch+1)+'.mat', mdict={'predict': predict_out,'labels':label_out})     
            print('Testing loss after %d Epochs: %f' %(epoch + 1, loss_test/count))
    print ('finished training')
    
    loss_test = 0.0
    count = 0
    label_out = np.array([])
    predict_out = np.array([])
    for i,data in enumerate(test_loader,0):
        images, labels = data
        images,labels = Variable(images.cuda()),Variable(labels.cuda())
        outputs = cnn(images)
        loss_te = loss(outputs,labels)
        loss_test += loss_te.data[0]
        if i==0:
            label_out = np.squeeze(np.array(labels.data))
            predict_out = np.squeeze(np.array(outputs.data))
        else:
            label_out = np.append(label_out,np.squeeze(np.array(labels.data)),axis=0)
            predict_out = np.append(predict_out,np.squeeze(np.array(outputs.data)),axis=0)
        count += 1
    scipy.io.savemat(result_dir[0]+'/test_real_27net1_'+str(epoch+1)+'.mat', mdict={'predict': predict_out,'labels':label_out})     
    print('Final Testing Loss: %f' %(loss_test/count))


# In[ ]:


class CNNModel(torch.nn.Module):
    def __init__(self,depth,depth_step):
        super(CNNModel,self).__init__()
        self.conv1 = torch.nn.Conv2d(n_channel, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm2d(depth)
        
        self.conv2 = torch.nn.Conv2d(depth, depth, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm2d(depth)
        
        self.conv3 = torch.nn.Conv2d(depth, depth, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.bn3 = torch.nn.BatchNorm2d(depth)
        
        depth = depth_step*depth
        
        self.conv4 = torch.nn.Conv2d(int(depth/depth_step), depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv4.weight)
        self.bn4 = torch.nn.BatchNorm2d(depth)
        
        self.conv5 = torch.nn.Conv2d(depth, depth, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv5.weight)
        self.bn5 = torch.nn.BatchNorm2d(depth)
        
        self.conv6 = torch.nn.Conv2d(depth, depth, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv6.weight)
        self.bn6 = torch.nn.BatchNorm2d(depth)
        
        depth = int(depth/depth_step)
        
        self.conv7 = torch.nn.Conv2d(depth*depth_step, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv7.weight)
        self.bn7 = torch.nn.BatchNorm2d(depth)
        
        self.conv8 = torch.nn.Conv2d(depth, depth, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv8.weight)
        self.bn8 = torch.nn.BatchNorm2d(depth)
        
        self.conv9 = torch.nn.Conv2d(depth, depth, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv9.weight)
        self.bn9 = torch.nn.BatchNorm2d(depth)
        
        depth = int(depth/(2*depth_step))
        
        self.conv10 = torch.nn.Conv2d(depth*2*depth_step, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv10.weight)
        self.bn10 = torch.nn.BatchNorm2d(depth)
        
        self.conv11 = torch.nn.Conv2d(depth, depth, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv11.weight)
        self.bn11 = torch.nn.BatchNorm2d(depth)
        
        self.conv12 = torch.nn.Conv2d(depth, depth, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv12.weight)
        self.bn12 = torch.nn.BatchNorm2d(depth)
        
        depth = int(depth/(2*depth_step))
        
        self.conv13 = torch.nn.Conv2d(depth*2*depth_step, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv13.weight)
        self.bn13 = torch.nn.BatchNorm2d(depth)
        
        self.conv14 = torch.nn.Conv2d(depth, depth, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv14.weight)
        self.bn14 = torch.nn.BatchNorm2d(depth)
        
        self.conv15 = torch.nn.Conv2d(depth, depth, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv15.weight)
        self.bn15 = torch.nn.BatchNorm2d(depth)
        
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p=0.5)
        
    def forward(self,data):
        x = self.drop(data)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn10(self.conv10(x)))
        x = self.relu(self.bn11(self.conv11(x)))
        x = self.relu(self.bn12(self.conv12(x)))
        x = self.relu(self.bn13(self.conv13(x)))
        x = self.relu(self.bn14(self.conv14(x)))
        x = self.relu(self.bn15(self.conv15(x)))
        return x

# In[ ]:


class largeCNN(torch.nn.Module):
    def __init__(self):
        super(largeCNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(n_channel, 2, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm2d(2)

        self.conv2 = torch.nn.Conv2d(2, 16, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm2d(16)

        self.conv2b = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2b.weight)
        self.bn2b = torch.nn.BatchNorm2d(16)   

        self.conv2c = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2c.weight)
        self.bn2c = torch.nn.BatchNorm2d(16)

        self.conv2d = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2d.weight)
        self.bn2d = torch.nn.BatchNorm2d(16)     

        self.conv2e = torch.nn.Conv2d(16, 2, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2e.weight)
        self.bn2e = torch.nn.BatchNorm2d(2)

        self.conv3 = torch.nn.Conv2d(2, 1, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.bn3 = torch.nn.BatchNorm2d(1)
        
        
        self.relu = torch.nn.ReLU()
        
    def forward(self,data):
        x = self.relu(self.bn1(self.conv1(data)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.relu(self.bn2c(self.conv2c(x)))
        x = self.relu(self.bn2d(self.conv2d(x)))
        x = self.relu(self.bn2e(self.conv2e(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

# In[ ]:


class smallCNN(torch.nn.Module):
    def __init__(self):
        super(smallCNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(n_channel, 2, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm2d(2)

        self.conv2 = torch.nn.Conv2d(2, 16, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm2d(16)

        self.conv2b = torch.nn.Conv2d(16, 2, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2b.weight)
        self.bn2b = torch.nn.BatchNorm2d(2)

        self.conv3 = torch.nn.Conv2d(2, 1, 3, stride=1, padding=1, bias=True)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.bn3 = torch.nn.BatchNorm2d(1)
        
        
        self.relu = torch.nn.ReLU()
        
    def forward(self,data):
        x = self.relu(self.bn1(self.conv1(data)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

# In[ ]:


class actualCNN(torch.nn.Module):
    def __init__(self,depth):
        super(actualCNN,self).__init__()

        self.conv1 = torch.nn.Conv2d(n_channel, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm2d(depth)

        self.conv2 = torch.nn.Conv2d(depth, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm2d(depth)

        self.conv2b = torch.nn.Conv2d(depth, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2b.weight)

        self.conv2c = torch.nn.Conv2d(depth, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2c.weight)
        self.bn2c = torch.nn.BatchNorm2d(depth)

        self.conv2d = torch.nn.Conv2d(depth, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2d.weight)

        self.conv2e = torch.nn.Conv2d(depth, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2e.weight)

        self.conv2f = torch.nn.Conv2d(depth, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2f.weight)
        self.bn2f = torch.nn.BatchNorm2d(depth)

        self.conv2g = torch.nn.Conv2d(depth, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2g.weight)

        self.conv2h = torch.nn.Conv2d(depth, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2h.weight)

        self.conv2i = torch.nn.Conv2d(depth, depth, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv2i.weight)

        self.conv3 = torch.nn.Conv2d(depth, 1, 5, stride=1, padding=2, bias=True)
        torch.nn.init.xavier_normal(self.conv3.weight)
        
        self.relu = torch.nn.ReLU()

    def forward(self,data):
        x = self.relu(self.bn1(self.conv1(data)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv2b(x))
        x = self.relu(self.bn2c(self.conv2c(x)))
        x = self.relu(self.conv2d(x))
        x = self.relu(self.conv2e(x))
        x = self.relu(self.bn2f(self.conv2f(x)))
        x = self.relu(self.conv2g(x))
        x = self.relu(self.conv2h(x))
        x = self.relu(self.conv2i(x))
        x = self.conv3(x)
        return x

# In[ ]:


class Block_down(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Block_down, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.bn3 = torch.nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class Block_up(torch.nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Block_up, self).__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(in_planes, planes, kernel_size=1, bias=False)
        torch.nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.deconv2 = torch.nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        torch.nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.deconv3 = torch.nn.ConvTranspose2d(planes, int(planes/self.expansion), kernel_size=1, bias=False)
        torch.nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = torch.nn.BatchNorm2d(int(planes/self.expansion))

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != int(planes/self.expansion):
            self.shortcut = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_planes, int(planes/self.expansion), kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(int(planes/self.expansion))
            )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.deconv1(x)))
        out = self.relu(self.bn2(self.deconv2(out)))
        out = self.bn3(self.deconv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetBlock(torch.nn.Module):
    def __init__(self, block1,block2, num_blocks, num_classes=10):
        super(ResNetBlock, self).__init__()
        self.in_planes = 64

        self.conv1 = torch.nn.Conv2d(n_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.xavier_normal(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block1, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block1, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block1, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block1, 512, num_blocks[3], stride=2)
        
        self.conv2 = torch.nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.xavier_normal(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm2d(512)
        
        self.in_planes = 512
        self.dlayer1 = self._make_layer1(block2, 512, num_blocks[2], stride=2)
        self.dlayer2 = self._make_layer1(block2, 256, num_blocks[2], stride=2)
        self.dlayer3 = self._make_layer1(block2, 128, num_blocks[1], stride=2)
        self.dlayer4 = self._make_layer1(block2, 64, num_blocks[0], stride=1)
        
        self.conv3 = torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=(2,1), bias=False)
        torch.nn.init.xavier_normal(self.conv3.weight)
        self.bn3 = torch.nn.BatchNorm2d(1)
        
        self.conv4 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.xavier_normal(self.conv4.weight)
        
        self.relu = torch.nn.ReLU()
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)
    
    def _make_layer1(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = int(planes/block.expansion)
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dlayer1(out)
        out = self.dlayer2(out)
        out = self.dlayer3(out)
        out = self.dlayer4(out)
        
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        
        return out

def ResNet():
    return ResNetBlock(Block_down,Block_up, [3,4,6,3])


# In[ ]:

model=config['model']

if(model[0]=='cnn1'):
    cnn = smallCNN()
elif(model[0]=='cnn2'):
    cnn = largeCNN()
elif(model[0]=='cnn0'):
    depth = n_channel
    cnn = actualCNN(depth)
    n_epochs,res_ep,eta = 200,50,0.0003
elif(model[0]=='cnn'):
    cnn = CNNModel(depth,depth_step)
elif(model[0]=='resnet'):
    cnn = ResNet()
    n_epochs,res_ep,eta = 20,4,0.0003
else:
    raise ValueError('Wrong model defined')
#cnn = CNNModel(depth,depth_step)
#cnn = smallCNN()
cnn.cuda()
train_test(cnn,eta,n_epochs,res_ep)

