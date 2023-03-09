'''
Origin dataset for Image2InChI is from Bristol-Myers Squibb
which is supported by Kaggle. https://www.kaggle.com/c/bms-molecular-translation/data
In this competition, there are images of chemicals, with the objective of
predicting the corresponding International Chemical Identifier (InChI) text string of the image.
The images provided (both in the training data as well as the test data) may be rotated to different angles,
be at various resolutions, and have different noise levels.
The sizes of images from original dataset are not fixed.
So here I will draw new molecules with 300*300 with RDKit to solve the problems of images of noise.

Datum: 19.03.2022
'''
import os.path
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm.auto import tqdm # solve the problem of each iteration of progressbar starts a new line

img_path_origin = './origin_img' # save images from original dataset
img_path_clear = './clear_img' #save new generated clear images
img_path_noise = './noise_img'#save new generated images with adding some noises.
img_path_clear_75 = './clear_img_smiles75+'#save new generated images with length of smiles sequence more than 75.

if not os.path.exists(img_path_origin):
    os.mkdir(img_path_origin)
else:
    pass

if not os.path.exists(img_path_clear):
    os.mkdir(img_path_clear)
else:
    pass

if not os.path.exists(img_path_noise):
    os.mkdir(img_path_noise)
else:
    pass

if not os.path.exists(img_path_clear_75):
    os.mkdir(img_path_clear_75)
else:
    pass

path = '/org/temp/anon/data/bms-molecular-translation_InChIs/'

df = pd.read_csv(path + "train_labels.csv")


head_inchi = ["file_name", "InChI"]
head_smiles = ["file_name", "SMILES"]

#store labels of inchi
label_inchi = open('train_inchi.csv', 'w')
label_inchi_writer = csv.writer(label_inchi)
label_inchi_writer.writerow(head_inchi)

#store labels of smiles
label_smiles = open('train_smiles.csv', 'w')
label_smiles_writer = csv.writer(label_smiles)
label_smiles_writer.writerow(head_smiles)


#store labels of smiles > 75
label_smiles_75 = open('train_smiles75+.csv', 'w')
label_smiles_75_writer = csv.writer(label_smiles_75)
label_smiles_75_writer.writerow(head_smiles)

#store inchi labels of smiles > 75
label_inchi_75 = open('train_inchi_smiles75+.csv', 'w')
label_inchi_75_writer = csv.writer(label_inchi_75)
label_inchi_75_writer.writerow(head_inchi)


#
'''
Draw molecules
'''
# read original image from dataset
img_num = 0
max_length = 75

for _, row in df.head(1000000).iterrows(): #Iterate over DataFrame rows as (index, Series) pairs.
    img_id = row['image_id']
    print('img_id:', img_id)

    #img: original image
    try:
        f = open(path + "train/{}/{}/{}/{}.png".format(img_id[0], img_id[1], img_id[2], img_id))

        img = cv2.imread(path + "train/{}/{}/{}/{}.png".format(img_id[0], img_id[1], img_id[2], img_id), cv2.IMREAD_GRAYSCALE)
        #Add check if the img exists? if not exists, skip, avoid error.
        #     print("os.path.exists(img)", os.path.exists(img))
        #     assert os.path.exists(img)

        #cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.
        #cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.
        #cv2.IMREAD_UNCHANGED: It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.
        # Could we use grayscale to make model focus on features of structure during training?

        #rename images id
        img_name = str(img_num) + '.png'
        print("img_name:", img_name)
        img_num += 1
        img_full_name_origin = os.path.join(img_path_origin, img_name)
        img_full_name_clear = os.path.join(img_path_clear, img_name)
        img_full_name_noise = os.path.join(img_path_noise, img_name)
        img_full_name_clear_75 = os.path.join(img_path_clear_75, img_name)

        print("img_full_name_origin:", img_full_name_origin)
        print(img_full_name_clear)
        print(img_full_name_noise)



        #draw new clearer molecule with 0 degree rotation
        InChI = row['InChI']
        mol_inchi = rdkit.Chem.inchi.MolFromInchi(row['InChI'])
        print("InChI sequence:", InChI)

        #draw molecule
        d = rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DCairo(300, 300)
        d.drawOptions().useBWAtomPalette()
        d.drawOptions().rotate = 0
        d.drawOptions().bondLineWidth = 1
        d.DrawMolecule(mol_inchi)
        d.FinishDrawing()
        d.WriteDrawingText("0.png")
        img_clear = cv2.imread("0.png", cv2.IMREAD_GRAYSCALE)


        '''
        Draw molecules with noise
        we can also draw some molecules with different level of noise.
        '''
        # molecule pixel coordinates
        y, x = np.where(img_clear < 250)

        # number of random pixels to replace (higher = more noise)
        n_samples = np.random.randint(len(y))

        # choose random pixel indices
        i = np.random.randint(0, len(y), n_samples)

        # replace pixels
        img_noisy = img_clear.copy()
        img_noisy[y[i], x[i]] = 260


        #print("Chem.MolToSmiles(Chem.MolFromInchi(InChI))", Chem.MolToSmiles(Chem.MolFromInchi(InChI), isomericSmiles=False,kekuleSmiles=True))


        #Generate the corresponding smiles sequence.
        #smiles = Chem.MolToSmiles(mol_inchi, isomericSmiles=False,kekuleSmiles=True)
        smiles = Chem.MolToSmiles(mol_inchi, canonical=True, isomericSmiles=False)
        smiles_iso = Chem.MolToSmiles(mol_inchi, canonical=False, isomericSmiles=True)
        print("Smiles:", smiles)
        inchi_list = [img_name, InChI]
        smiles_list = [img_name, smiles]

        if len(smiles) <= max_length:
            label_inchi_writer.writerow(inchi_list)

            label_smiles_writer.writerow(smiles_list)

            #change the size of img into 300*300, original dataset
            #img = cv2.resize(img, (300, 300))
            cv2.imwrite(img_full_name_origin, img)
            cv2.imwrite(img_full_name_clear, img_clear)
            cv2.imwrite(img_full_name_noise, img_noisy)
        else:
            label_inchi_75_writer.writerow(inchi_list)
            label_smiles_75_writer.writerow(smiles_list)

            #change the size of img into 300*300, original dataset
            #img = cv2.resize(img, (300, 300))
            cv2.imwrite(img_full_name_clear_75, img_clear)



    #print("Isomeric Smiles:", smiles_iso)

    except IOError:
        print("Image file " + img_id +" not accessible")
    finally:

        #label_inchi.close()
        # checking the completion
        if img_num % 1000 == 0 :
            print('finish index : {1}'.format(img_num))

label_inchi.close()
label_smiles.close()
label_inchi_75.close()
label_smiles_75.close()









