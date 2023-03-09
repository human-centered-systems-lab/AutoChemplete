'''
Generate image2smiles dataset from RDKit according to inchi sample.
generate only 2 dimensions image
generate different level of noise.


'''

import os.path
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import cv2



#generate data for test
# df = pd.read_csv("/org/temp/anon/data/RDkit_SMILES_gray/test_RDKit_labels.csv")
# img_path = '/org/temp/anon/data/RDkit_SMILES_gray/test_clear' # save new images
# img_path_noise = '/org/temp/anon/data/RDkit_SMILES_gray/test_noise'#save new generated images with adding some noises.

# #generate data for train
# df = pd.read_csv("/org/temp/anon/data/RDkit_SMILES_gray/train_RDKit_labels.csv")
# img_path = '/org/temp/anon/data/RDkit_SMILES_gray/train_clear' # save new images
# img_path_noise = '/org/temp/anon/data/RDkit_SMILES_gray/train_noise'#save new generated images with adding some noises.

# #generate data for train

df = pd.read_csv("/org/temp/anon/data/new_images_5M_75_noise/train.csv")
#img_path = '/org/temp/anon/data/new_images5M_75_noise/train' # save new images
img_path_noise = '/org/temp/anon/data/new_images_5M_75_noise/train'#save new generated images with adding some noises.

print(img_path_noise)
if not os.path.exists(img_path_noise):
    os.mkdir(img_path_noise)
else:
    pass

# if not os.path.exists(img_path):
#     os.mkdir(img_path)
# else:
#     pass


file_writer = open("/org/temp/anon/data/new_images_5M_75_noise/train_new.csv", 'w')
file_writer.write("file_name,SMILES"+"\n")


'''
Draw molecules
'''
# read original image from dataset
img_num = 0
print("start")
for _, row in df.iterrows(): #Iterate over DataFrame rows as (index, Series) pairs.
    idx = row['file_name']
    img_num += 1
    smiles = row['SMILES']
    #print('SMILES', smiles)
    try:
        mol = Chem.MolFromSmiles(smiles)
        #smiles = row['']

        #draw molecule
        d = rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DCairo(300, 300)
        d.drawOptions().useBWAtomPalette()
        d.drawOptions().rotate = 0
        d.drawOptions().bondLineWidth = 1
        d.DrawMolecule(mol)
        d.FinishDrawing()
        d.WriteDrawingText("0.png")

        #img_clear = cv2.imread("0.png", cv2.IMREAD_COLOR)
        img_clear = cv2.imread("0.png", cv2.IMREAD_GRAYSCALE)

        #img_full_name_clear = os.path.join(img_path, idx)
        #cv2.imwrite(img_full_name_clear, img_clear)

        #Draw molecules with noise
        y, x = np.where(img_clear < 240) # molecule pixel coordinates

        # number of random pixels to replace (higher = more noise)
        n_samples = np.random.randint(len(y))

        # choose random pixel indices
        i = np.random.randint(0, len(y), n_samples)

        # replace pixels
        img_noisy = img_clear.copy()
        img_noisy[y[i], x[i]] = 240

        img_full_name_noise = os.path.join(img_path_noise, idx)
        cv2.imwrite(img_full_name_noise, img_noisy)
        file_writer.write(idx + "," + smiles + "\n")


    except IOError:
        print("Image file " + idx +" not accessible")
    finally:
        # checking the completion
        if img_num % 1000 == 0 :
            print('finish index : {0}'.format(img_num))