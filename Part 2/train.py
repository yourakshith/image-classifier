import argparse
import helper as hp

import torch
import torch.nn as nn
from torch import optim
from torchvision import models

import time

parser = argparse.ArgumentParser(description = "Train the neural network")

parser.add_argument("data_dir", help="Path for Image dataset. Check your present working directory while providing this argument.",type=str)
parser.add_argument("--save_dir", help="Location of where the checkpoint should be saved. Check your present working directory while providing this argument.",default="/home/workspace/ImageClassifier/checkpoint.pth",type=str)
parser.add_argument("--arch", help="Architecture used to train the model (ResNet only!)\nWe are using only ResNet models as it's fast and   saving model takes less space and the now the code part has model.fc instead of model.classifier... so use ResNet(recommended) or any arch which has model.fc", default='resnet152',type=str)
parser.add_argument("--lr", help="Learning rate of the model",default=0.001,type=float)
parser.add_argument("--hidden", help="No. of hidden units in the model",nargs="+",default=[256,126],type=int)
parser.add_argument('--dropout', help='Dropout during learning', type=float, default=0.1)
parser.add_argument('--epochs', help='Number of epochs for training', default=5, type=int)
parser.add_argument('--gpu', help='Enable GPU', action='store_true', default=False)

#Convert argument parameters into variables
args = parser.parse_args()

data_dir = args.data_dir
checkpoint_dir = args.save_dir
arch=args.arch
learning_rate = args.lr
hidden_units = args.hidden
dropout = args.dropout
epochs = args.epochs
gpu = args.gpu

train_dir = data_dir+'/train'
test_dir = data_dir+'/test'
valid_dir = data_dir+'/valid'

trainloader, testloader, validationloader, class_to_idx = hp.load_transformed_data(train_dir,test_dir,valid_dir)

#train
model = getattr(models,arch)(pretrained=True)
for param in model.parameters():
    param.requires_grad=False
input_size = model.fc.in_features
output_size = len(class_to_idx)
model.fc = hp.BuildNetwork(input_size,output_size,hidden_units,dropout)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() and gpu==True else 'cpu')

model = hp.train(model,device,epochs,criterion,optimizer,trainloader,validationloader)

loss, acc = hp.validate(model,device,criterion,testloader)

print(f"Test loss: {loss:.3f}\nTest Accuruacy: {acc:.3f}%")

save = checkpoint_dir
flag = hp.save_model(arch,model,optimizer,criterion,class_to_idx,save)
if flag:
    print(f"Model saved as checkpoint successfully at {save}")
else:
    print("Try saving model again")