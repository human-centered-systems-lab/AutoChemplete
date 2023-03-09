"""
The code below is the code that divides about 110 million
total data frames into groups according to the number of cores
for parallel processing and generates data frames for each group.
The generated group-by-group data frames are used as inputs for
the following files: train_image_generation.py.
"""


import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm
import math
import os

# CID_SMILES is downloaded from pubchem
# Uploading the data
f = open("/home/anon/data/CID-SMILES")

# The number of total data
#len(list(f)) #110604442

# Removing unneccesary charaters & Extracting SMILES sequences

a = []
usage = []
length = []

#choose how much sequence we want to extract.
#14% of 1 million needs 200gb to store .npy files, very lagre and needs long time.

#Even if only data with a length of 100 or less were generated previously, it was still a large amount of data,
# so it was impossible to use all of them for training.
#we extract 5 million as backup

n_images = 20000000 #20M
print("SMILES in total:".format(n_images))
new_path = 'train_dataset_20M/'

#14% of 1 million needs 200gb to store .npy files, very lagre and needs long time.

if os.path.exists(new_path) == False:
    os.mkdir(new_path)
else:
    pass


#choose the length less than 100
max_length = 100
count = 0

for i in tqdm(range(n_images)):
    line = f.readline()
    if line ==  "": break
    # a.append(line)
    # if len(line[:-1].split("\t")) > 1:
    #    a[i] = a[i][:-1].split("\t")[1]
    # usage.append(a[i])
    # length.append(len(a[i]))
    parts = line.strip().split()

    if len(parts) > 1:
        sequence = parts[1]
        if len(sequence) <= max_length:
            usage.append(sequence)
            count += 1
    else:
        print("Error! sequence not found")
        print(i)
        print(line)
#        exit()
# Making dataframe
df = pd.DataFrame(usage)
df.columns = ["SMILES"]
#df['length'] = length
df['group'] = 0

# Oragainizing the group by the number of core
# The number of data sameple for one group calculated as
# the number of total data sample / the number of core
# ex) 111307682 / 31 = 3700000
# The number of core can be different by each environment

#here: 5M / 5 =  1M
'''
for i in range(1, 32) : 
    filtered_df = df
    filtered_df['group'][(i-1)* 3700000 : i * 3700000] = i

new_path = '/train_dataset/'
for i in range(1, 32) :
    g_filtered = filtered_df[filtered_df['group'] == i]
    g_filtered.to_csv(new_path + "filtered_df_group{}.csv".format(i)) 
'''

n_groups = 4

group_size = math.ceil(count / n_groups)
#mat.ceil:
for i in range(1, n_groups+1) :
    filtered_df = df
    filtered_df['group'][(i-1)* group_size : i * group_size] = i



for i in range(1, n_groups+1) :
    g_filtered = filtered_df[filtered_df['group'] == i]
    g_filtered.to_csv(new_path + "filtered_df_group{}.csv".format(i))