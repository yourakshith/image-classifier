import argparse
import torch
import json
import helper as hp

parser = argparse.ArgumentParser(description = "Image Classifier for Flowers")

parser.add_argument("path",help="Path of the image for prediction. Check your present working directory while providing this argument.",type=str)
parser.add_argument("checkpoint",help="Path to the checkpoint. Check your present working directory while providing this argument.",type=str)
parser.add_argument("--top_k",default=5,help="Display the top-k classes of predicition",type=int)
parser.add_argument("--category_names",default="/home/workspace/ImageClassifier/cat_to_name.json",help="File mapping categories with real names. Check your present working directory while providing this argument.",type=str)
parser.add_argument('--gpu', help='Enable GPU', action='store_true', default=False)

args = parser.parse_args()

image = args.path
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

model, optimizer, criterion = hp.load_model(checkpoint)
device = torch.device('cuda' if torch.cuda.is_available() and gpu==True else 'cpu') 

probs,classes = hp.predict(image, model, top_k, device, cat_to_name)

print(f"\nPrediction: {classes[0]} ({probs[0]:.3f})\n")

print(f"The top {top_k} classes and their corresponding probabilites are:")
for i,j in zip(classes, probs):
    print(f"{i} ({j:.3f})")