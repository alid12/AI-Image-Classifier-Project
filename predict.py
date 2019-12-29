import matplotlib.pyplot as plt
import numpy as np


import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
import json
from collections import OrderedDict
from PIL import Image


from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import time
import utils

ap = argparse.ArgumentParser(description='Predict.py')


ap.add_argument('--input', default='./flowers/test/1/image_06752.jpg', action="store", type = str)

ap.add_argument('--checkpoint', default='./checkpoint.pth', action="store", type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input
num_outputs = pa.top_k
device = pa.gpu
file=pa.category_names
path = pa.checkpoint
#pa = ap.parse_args()

def main():
    model=utils.load_checkpoint(path)
    with open(file, 'r') as f:
        cat_to_name = json.load(f) 
    probabilities = utils.predict(path_image, model, num_outputs, device)
    
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in probabilities[1][0].tolist()]
    labels = [cat_to_name[cat] for cat in classes]

    probability = np.array(probabilities[0][0])
    i=0
    while i < num_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Predicting is over")

    
if __name__== "__main__":
    main()

