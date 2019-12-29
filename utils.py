
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
from PIL import Image
import torchvision.models as models
from torch import tensor
import argparse
from torch.autograd import Variable


import json
structures = {"vgg16":25088,
         "densenet121":1024,
         "alexnet":9216}

def  transform_image(root):
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms={
        'train_transforms': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),  
        'test_transforms': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
        'validation_transforms': transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets={
        'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
        'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['validation_transforms']),
        'test_data' : datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms'])
    } 
    return image_datasets['train_data'] , image_datasets['valid_data'], image_datasets['test_data']


def load_data(root):
    data_dir = root    
    train_data,valid_data,test_data=transform_image(data_dir)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders ={
        'trainloader':torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True),
        'validloader': torch.utils.data.DataLoader(valid_data, batch_size=32),
        'testloader' : torch.utils.data.DataLoader(test_data, batch_size=32)
    } 
    
    return dataloaders['trainloader'],dataloaders['validloader'],dataloaders['testloader']
     
                                                            
                                  
def network (structure='vgg16',dropout=0.5, hidden_layer1 = 4096,lr = 0.001, device='gpu'):
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print ("Please use either vgg16, densenet121, or alexnet structures as {} is not a valid one." .format(structure))
        
    for param in model.parameters():
        param.requires_grad = False

    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(structures[structure],hidden_layer1)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_layer1, 1024)),
        ('relu2', nn.ReLU()), 
        ('dropout2', nn.Dropout(dropout)),
        ('fc3', nn.Linear(1024, 102)),
        ('output', nn.LogSoftmax(dim=1))
         ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr) 
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()
   
    return model, criterion , optimizer

def train_network(model, criterion, optimizer, epochs = 3, print_every=40,tloader=0,vloader=0, device='gpu'):
                                                            
    
    steps = 0
   
    for e in range(epochs):
        running_loss = 0
    
        for ii, (inputs, labels) in enumerate(tloader):
            steps += 1
            if torch.cuda.is_available() and device =='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                loss = 0
                accuracy=0
            
            
                for ii, (inputs2,labels2) in enumerate( vloader):
                
                    optimizer.zero_grad()
                    if torch.cuda.is_available() and device == 'gpu':
                        inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda')
                        model.to('cuda')
                
                    with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            loss = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()
                loss = loss / len(vloader)
                accuracy = accuracy /len(vloader)
            
            
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(loss),
                       "Accuracy: {:.4f}".format(accuracy))
            
            
                running_loss = 0           
def save_checkpoint(model,traindata,path='checkpoint.pth',structure ='vgg16', hidden_layer1 = 4096,dropout=0.5,lr=0.001,epochs=3):
    # TODO: Save the checkpoint 
    model.class_to_idx = traindata.class_to_idx
    model.cpu
    torch.save({'structure' :structure,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path='checkpoint.pth'): 
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']                                                        
                                                            
    lr=checkpoint['lr']
    model,_,_ = network(structure , dropout,hidden_layer1,lr)
    
    model.load_state_dict(checkpoint['state_dict'])    
    model.class_to_idx = checkpoint['class_to_idx']  
    return model
                                                           
                                                           
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    np_image = Image.open(image)
    processor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    # TODO: Process a PIL image for use in a PyTorch model
    processed_image = processor(np_image)    
    return processed_image

                                                            
def predict(image_path, model, topk=5, device='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if torch.cuda.is_available() and device =='gpu':
        model.to('cuda')
    else:
        model.to('cpu')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if torch.cuda.is_available() and device =='gpu':                                                        
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)                                                        
    
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)                                                  


    
