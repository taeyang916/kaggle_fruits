from __future__ import print_function, division
import os
import torch
import csv
import pandas as pd
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils

root_path = "/fruits-360"
Training_path = "/Training"
Test_path = "/Test"

"""
workspace_path = os.path.dirname(os.path.realpath(__file__))
dir_path = workspace_path + root_path + Training_path

os.listdir(dir_path)

print("before: %s"%os.getcwd())
os.chdir("./ty" + root_path + Training_path)
print("after: %s"%os.getcwd())

test_dir = os.listdir(os.getcwd())
#test_dir.sort()
print(test_dir)

for i in test_dir:
    print(i)
    test_dir += 'i'
"""

data = [{'dataname':[], 'idx':[]}]
class FruitDataLoader(DataLoader):
    def __init__(self, root='./fruits-360', transform=False):
        super.__init__(FruitDataLoader)



        datasets_list = os.listdir(os.getcwd()).sort()

    def __getitem__(self, idx):
        
        return
    
    def __len__(self):

        return
