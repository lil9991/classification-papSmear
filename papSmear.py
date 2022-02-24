#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:25:21 2021

@author: argenit
"""

#%%
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np 
import glob
import os

from datetime import timedelta
#from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer
from scipy.io import savemat
from PIL import Image
from sklearn.metrics import f1_score
import pretrainedmodels

#%%

#data_loader_with_paths

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


#%%
#mean, std

mean = 0
std = 0
nb_samples = 0

m_std_transforms = transforms.Compose([  
   transforms.Resize((299,299)),
   transforms.ToTensor()])

m_std_dataset = datasets.ImageFolder('Train', transform= m_std_transforms)
m_std_loader = torch.utils.data.DataLoader(m_std_dataset, batch_size = 32, shuffle=True)

for data,_ in m_std_loader:
         batch_samples = data.size(0)
         data = data.view(batch_samples, data.size(1), -1)
         mean += data.mean(2).sum(0)
         std += data.std(2).sum(0)
         nb_samples += batch_samples
    
mean /= nb_samples
std /= nb_samples
print(mean)
print(std)

# %%
#resnet18 #resnet101
from resnet_pytorch import ResNet
model = ResNet.from_pretrained('resnet101', num_classes=5)
#%%

#model1-1

model = pretrainedmodels.xception(num_classes=1000, pretrained='imagenet')
model.fc = torch.nn.Linear(1000, 5)
#%%

model = models.mobilenet_v2(pretrained=True)
model.fc = torch.nn.Linear(1280, 5)

#%%

#model3-1

#model = models.resnext50_32x4d(pretrained=True) 
#model.fc = torch.nn.Linear(2048, 5)

#%%
#resnet50
#import torchvision.models as models
#model = models.resnet50(pretrained=True)
#model.fc = torch.nn.Linear(2048, 5)

#%%
#densenet 121/169/201

#from densenet_pytorch import DenseNet
#model = DenseNet.from_pretrained("densenet121")
#model.fc = torch.nn.Linear(1000, 2)

#%%
#efficientnetcb0/b1/b2/b3/b4

#from efficientnet_pytorch import EfficientNet
#model = EfficientNet.from_pretrained('efficientnet-b4') 
#model.fc = torch.nn.Linear(1000, 5)



# %%
# data augmentation

train_transforms = transforms.Compose([
            #transforms.Grayscale(num_output_channels=3),
             transforms.Resize((299,299)),
             transforms.RandomRotation(degrees = (0, 180)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean = mean, std = std) ])


test_transforms = transforms.Compose([
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std) ])


#%%
#train_dataset

batch_size=32
train_dataset = datasets.ImageFolder('Train', transform= train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True) 

print('Train Dataset: ', len(train_dataset))


#%%                                         
#test_dataset
# instantiate the dataset and dataloader

#test_dataset = ImageFolderWithPaths("Dataset/Test", transform= test_transforms)
test_dataset = datasets.ImageFolder('Test', transform= train_transforms) 
test_loader = torch.utils.data.DataLoader(test_dataset,  batch_size, shuffle = False)

print("Test Dataset: " , len(test_dataset))


#%%
#1
#optimization Adam
#criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
#%%
#2
#optimizasyon SGD
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

#%%
import torch
torch.cuda.empty_cache()


#%%

total_epoch = 50
best_acc = 0
best_acc = 0
input_size = 0
acc_vect = []
cm_vect = []
f1score_vect = []


model = model.cuda().train()

for epoch in range(total_epoch):
    running_loss = 0
    iterx = 0    

    #Train 
   
    for X, y in train_loader: 
        
        X = X.cuda()
        y = y.cuda()
        
        optimizer.zero_grad()
        
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data
        
       
    # Test ve Confusion Matrix
    model.eval()
    
    total = 0
    true = 0
    fn = 0
    fp = 0 
    pre_vect = np.array([])
    gnd_vect = np.array([])
   
    fp_paths = []
    fn_paths = []
    
    
    with torch.no_grad():
        for X, y in test_loader: #, pths

            X = X.cuda()
            y = y.cuda()

            output = model(X)

            _, maksimum = torch.max(output.data, 1)

            pre_ = maksimum.detach().cpu().numpy()
            gnd_ = y.detach().cpu().numpy()
            
             
            
            #if(pre_ == 0 and gnd_ == 1):
             #   fn+=1
             #   fn_paths.append(pths)  
               
            #elif(pre_== 1 and gnd_ == 0):
               #   fp+=1
             #   fp_paths.append(pths)
                

            pre_vect = np.r_[pre_vect, pre_]
            gnd_vect = np.r_[gnd_vect, gnd_]

            total += y.size(0)
            true += (maksimum == y).sum().item()
            accuracy = true/total
            
     #list_acc_cm
    acc_vect.append(accuracy)
    cm_vect.append(confusion_matrix(gnd_vect, pre_vect))
    f1score_vect.append(f1_score(gnd_vect, pre_vect, average= 'macro'))
    
    
    #save_model 
    if accuracy >= best_acc:
        torch.save(model.state_dict(), "xception-model_best.pth")
        best_acc = accuracy

    torch.save(model.state_dict(), "xception-model_latest.pth")

    print("Epoch:", epoch, ", Accuracy:", accuracy *100, ",  total_loss:", running_loss)


end = timer()




# %%

#print("fn: {}, fp: {} ".format(fn,fp))

#%%

#Acuracy, Time
print("Max Accuracy: {:.2f}".format(best_acc*100))
#print('\nTime: ', timedelta(seconds=end-start))

#%%

mean_acc = np.mean(acc_vect)
print("Accuracy mean: ", mean_acc)


#%%
# def save(path, num):
#     #dir_l = fn_paths[3] 
#     str_path = ''.join(path)
#     diir = os.path.split(str_path) #save için 
#     im = Image.open(str_path)
#     im2 = im.copy()
#     #save("./fp/"+diir[1]+"_"+str(num)+".bmp")
#     #im3 = im2.save("./fn-fp/resnet18/fn/" +diir[1]+"_"+str(num)+"_"+".bmp")
#     im3 = im2.save("./fn-fp/resnet101/fp/" +diir[1]+".bmp")

#%%
# num = 1

# for filename in fp_paths:
#     save(filename, num)
#     num += 1


#%%

CM = confusion_matrix(gnd_vect, pre_vect)
print("\nConfusion Matrix:  \n\n",CM)


#%%
# %'lik oran
# import seaborn as sns
# sns.heatmap(CM/np.sum(CM), annot=True, 
#             fmt='.2%', cmap='Blues')

#%%

# sayısal değer
import scikitplot.metrics as splt  
splt.plot_confusion_matrix(gnd_vect, pre_vect)


#%%
CM = confusion_matrix(gnd_vect, pre_vect)
print("\nConfusion Matrix:  \n\n",CM)

def statconf(CM):
    
    tot_sample  = np.sum(CM)
    L = len(CM)    
    acc = np.trace(CM) / tot_sample
    pre = np.zeros(L)
    rcl = np.zeros(L)
    f1s = np.zeros(L)
    
    for i in range(len(CM)):
        pre[i] = CM[i,i]/np.sum(CM[:,i])
        rcl[i] = CM[i,i]/np.sum(CM[i,:])
        f1s[i] = 2*pre[i]*rcl[i] / ( pre[i] + rcl[i] )
        
    return (acc , np.mean(pre) , np.mean(rcl) , np.mean(f1s))

xc = statconf(CM)
 
print("\n\n Accuracy : {:.4f} , Precision : {:.4f} , Recall : {:.4f} , F1-Score : {:.4f}".format( xc[0] , xc[1] , xc[2] ,  xc[3]))

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
print('recall: %.4f' % recall_score(gnd_vect, pre_vect, average= 'macro'))
print('precision: %.4f' % precision_score(gnd_vect, pre_vect, average= 'macro'))
print('F1 Score: %.4f' % f1_score(gnd_vect, pre_vect, average= 'macro'))
print('acc: %.4f' % accuracy_score(gnd_vect, pre_vect))


# %%
#save Confusion matrix, accuracy

#gnd   = {"gnd_vect":gnd_vect, "label":"gnd Vector"}
#pre   = {"Pre_vect": pre_vect,"label":"Pre Vector"}
f1   = {"f1_score": f1score_vect,"label":"F1_score Vector"}
ac = {"accuracy": acc_vect, "label": "accuracy vector"}
cm = {"CM": cm_vect, "label": "confusion matrix"}
savemat("xception-model.mat", {"Accuracy Vector": ac,"F1_Score ": f1 , "Confusion Matrix": cm})


# %%

#load mat_file
mat_file = sio.loadmat("resnext50-model.mat")
print(mat_file)


#%%
