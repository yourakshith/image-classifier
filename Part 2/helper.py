import torch
import torch.nn as nn
from torchvision import transforms, datasets, models

import numpy as np
import time
import json

from PIL import Image
from collections import OrderedDict
"""
Functions defined in helper:
--> BuildNetwork
--> load_transformed_data
--> train
--> validate
--> save_model
--> load_model
--> process_image
--> predict
"""
def BuildNetwork(input_size,output_size,hidden_units,dropout):
    d = OrderedDict()
    d['fc1'] = nn.Linear(input_size,hidden_units[0])
    d['relu1']=nn.ReLU()
    d['dropout1']=nn.Dropout(dropout)
    c=2
    for i in range(len(hidden_units)-1):
        d['fc'+str(c)] = nn.Linear(hidden_units[i],hidden_units[i+1])
        d['relu'+str(c)]=nn.ReLU()
        d['dropout'+str(c)]=nn.Dropout(dropout)
        c+=1
    d['output'] = nn.Linear(hidden_units[-1],output_size)
    d['softmax'] = nn.LogSoftmax(dim=1)
    return nn.Sequential(d)

def load_transformed_data(train_dir,test_dir,valid_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir,transform=validation_transforms)
    test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32,shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    
    return trainloader,testloader,validationloader,train_datasets.class_to_idx

def train(model,device,epochs,criterion,optimizer,trainloader,validationloader):
    model.to(device)
    start = time.time()
    print("Training on train dataset and validation test on valid dataset")
    print("Training started. Total epochs =",epochs)
    for epoch in range(epochs):
        tr_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device),labels.to(device)

            logps = model(images)

            optimizer.zero_grad()
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()

            tr_loss+=loss.item()
        else:
            #Making less validation steps to save training time
            #for-else instead of (if(steps%print_every==0)) condition
            #Validates after every epoch
            va_loss, acc = validate(model,device,criterion,validationloader)
            tr_loss = tr_loss/len(trainloader)
            print(f"Epoch {epoch+1}/{epochs}: "
                    f"Train loss: {tr_loss:.3f} "
                    f"Validation loss: {va_loss:.3f} "
                    f"Validation Accuracy: {acc:.3f}%")
            tr_loss = 0
            model.train()
    print("Training complete.")
    end=time.time()
    print(f"Total time: {(end-start):.2f} seconds.")
    return model

def validate(model,device,criterion,data):
    acc = 0
    loss = 0
    tot = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device),labels.to(device)
            log_ps = model(images)
            b_loss = criterion(log_ps,labels)
            loss += b_loss.item()
            ps = torch.exp(log_ps)
            top_p, top_cls = ps.topk(1, dim=1)
            equals = top_cls == labels.view(*top_cls.shape)
            acc += torch.sum(equals.type(torch.FloatTensor)).item()
            tot += labels.size(0)
        acc = acc/tot*100
    model.train()
    return loss, acc

def save_model(arch,model,optimizer,criterion,class_to_idx,save):
    checkpoint = {
    'arch': arch,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'class_to_idx': class_to_idx,
    'fc': model.fc,
    'criterion': criterion
    }

    torch.save(checkpoint, save)
    return True

def load_model(path): #path -> path of checkpoint
    checkpoint = torch.load(path)
    
    arch = checkpoint['arch']
    fc = checkpoint['fc']
    
    model = getattr(models,arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = fc
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer_dict']
    criterion = checkpoint['criterion']
    
    return model, optimizer, criterion

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    with Image.open(image) as im:
        width, height = im.size
        aspect_ratio = width/height
        target_size = 256
        im.resize((int(target_size),int(target_size*aspect_ratio))) #keeping the aspect ratio same along with target_size

        width, height = im.size
        ratio = width/height
        if width<=height:
            im.resize((target_size,int(ratio*target_size)))
        else:
            im.resize((int(ratio*target_size),target_size))
        
        crop = 224
        w1 = (im.size[0]-crop)/2
        w2 = (im.size[0]+crop)/2
        h1 = (im.size[1]-crop)/2
        h2 = (im.size[1]+crop)/2
        im = im.crop((w1,h1,w2,h2))

        arr = np.array(im)/255
        mean = np.array([0.485,0.456,0.406])
        sd = np.array([0.229,0.224,0.225])

        np_img = (arr-mean)/sd

        np_img = np_img.transpose((2,0,1))

    return np_img

def predict(image_path, model, topk, device, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    np_arr = process_image(image_path)
    ts = torch.from_numpy(np_arr)
    model.to(device)
    model.eval()
    with torch.no_grad():
        ts = (ts.float().to(device))
        ps = torch.exp(model(ts.unsqueeze(0)))
        probs, classes = ps.topk(topk)
        idx_to_cls={val:key for key,val in model.class_to_idx.items()}
        top_cls = [idx_to_cls[cls] for cls in classes.tolist()[0]]

        names = []

        for name in top_cls:
            names.append(cat_to_name[name])

    return probs.tolist()[0], names
