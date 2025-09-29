import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score
import numpy as np

### Loading data

import kagglehub
import os

# Download latest version of the dataset
path = kagglehub.dataset_download("mostafaabla/garbage-classification")
print("Path to dataset files:", path)

## List the folders, within the main folder
folders = os.listdir(path+'/garbage_classification')

print(folders)
print(len(path+'/garbage_classification/paper/'))


##### Load data

