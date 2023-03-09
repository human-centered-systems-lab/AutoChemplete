
#
import rdkit
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
import click

import warnings
warnings.filterwarnings(action = 'ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
#
'''#generate test_100K_labels.csv/ test_img_100K'''
# # path
# path_all = '/org/temp/anon/data/testset_isomeric/' # Saving new data
# if not os.path.exists(path_all):
#     os.mkdir(path_all)
# else:
#     pass
#
#
# path = path_all + '/test_img_100K/' # Saving new image
#
# data_path = '/org/temp/anon/data/train_dataset_10M/'
# if not os.path.exists(path):
#     os.mkdir(path)
# else:
#     pass
#
# file_writer = open(path_all + "test_100K_labels.csv", 'w')
# file_writer.write("file_name,SMILES"+"\n")
#
# @click.command()
# @click.option('--group', default=1, help='group number')
#
# def making_data(group):
#     print("group number:", group)
#     count = 0
#     filtered_df = pd.read_csv(data_path +'/filtered_df_group{}.csv'.format(group))
#     data_len = len(filtered_df)
#
#     for idx in range(data_len):
#         smiles = filtered_df['SMILES'][idx]  # this is the representation string
#         if len(smiles) <= 75:
#             count += 1
#             img_name = str(idx) + ".png"
#             smiles_g = Chem.MolFromSmiles(smiles)
#             try:
#                 # smile_plt is the image so we can directly save it.
#                 smile_plt = Draw.MolToImage(smiles_g, size = (300,300))
#
#                 img_full_name = os.path.join(path, img_name)
#                 file_writer.write(img_name + "," + smiles + "\n")
#                 smile_plt.save(img_full_name)  # save the image in png
#                 assert len(smiles) <= 75
#                 del (smile_plt)
#             except ValueError:
#                 pass
#         else:
#             pass
#
#         if count >= 100000: #only get 100K samples for test
#             break
#
#
#         # checking the completion
#         if idx % 1000 == 0 :
#             print('group : {0}, index : {1}'.format(group, idx))
#     print("Number of length <=75 is {0}".format(count))
#     del(filtered_df)
#     file_writer.close()
#
# if __name__ == '__main__':
#     making_data()



'''#generate test_100K_75to100_labels.csv and test_img_75to100_100K'''
# path
path_all = '/org/temp/anon/data/testset_isomeric/' # Saving new data
if not os.path.exists(path_all):
    os.mkdir(path_all)
else:
    pass


path = path_all + '/test_img_75to100_100K/' # Saving new image

data_path = '/org/temp/anon/data/train_dataset_10M/'
if not os.path.exists(path):
    os.mkdir(path)
else:
    pass

file_writer = open(path_all + "test_100K_75to100_labels.csv", 'w')
file_writer.write("file_name,SMILES"+"\n")

@click.command()
@click.option('--group', default=1, help='group number')

# def making_data(group):
#     print("group number:", group)
#     count = 0
#     filtered_df = pd.read_csv(data_path +'/filtered_df_group{}.csv'.format(group))
#     data_len = len(filtered_df)
#
#     for idx in range(data_len):
#         smiles = filtered_df['SMILES'][idx]  # this is the representation string
#         if len(smiles) > 75 and len(smiles) <= 100:
#             count += 1
#             img_name = str(idx) + ".png"
#             smiles_g = Chem.MolFromSmiles(smiles)
#             try:
#                 # smile_plt is the image so we can directly save it.
#                 smile_plt = Draw.MolToImage(smiles_g, size = (300,300))
#
#                 img_full_name = os.path.join(path, img_name)
#                 file_writer.write(img_name + "," + smiles + "\n")
#                 smile_plt.save(img_full_name)  # save the image in png
#                 assert len(smiles) > 75
#                 assert len(smiles) <= 100
#                 del (smile_plt)
#             except ValueError:
#                 pass
#         else:
#             pass
#
#         if count >= 100000: #only get 100K samples for test
#             break
#
#
#         # checking the completion
#         if idx % 10000 == 0 :
#             print('group : {0}, index : {1}'.format(group, idx))
#     print("Number of length >75, <=100 is {0}".format(count))
#     del(filtered_df)
#     file_writer.close()

#from multi groups
def making_data(group):
    for i in range(4):
        print("group number:", group)

        filtered_df = pd.read_csv(data_path +'/filtered_df_group{}.csv'.format(group))
        data_len = len(filtered_df)
        print("data length of this group:", data_len)
        group += 1
        count = 0
        for idx in range(data_len):
            smiles = filtered_df['SMILES'][idx]  # this is the representation string
            if len(smiles) > 75 and len(smiles) <= 100:
                count += 1
                img_name = str(idx) + ".png"
                smiles_g = Chem.MolFromSmiles(smiles)
                try:
                    # smile_plt is the image so we can directly save it.
                    smile_plt = Draw.MolToImage(smiles_g, size = (300,300))

                    img_full_name = os.path.join(path, img_name)
                    file_writer.write(img_name + "," + smiles + "\n")
                    smile_plt.save(img_full_name)  # save the image in png
                    assert len(smiles) > 75
                    assert len(smiles) <= 100
                    del (smile_plt)
                except ValueError:
                    pass
            else:
                pass
            if idx % 10000 == 0 :
                print('group : {0}, index : {1}'.format(group-1, idx))
        if count >= 10: #only get 100K samples for test
            break
        print("Number of length >75, <=100 is {0}".format(count))

    del(filtered_df)
    file_writer.close()

if __name__ == '__main__':
    making_data()