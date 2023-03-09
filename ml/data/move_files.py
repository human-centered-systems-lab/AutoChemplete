import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
# input_file = '/org/temp/anon/data/new_images_5M/train.csv'
# input_dir = '/org/temp/anon/data/RDkit_SMILES_gray/clear_img/'
#
# output_dir = "test_RDKit"
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
# else:
#     pass
#
# i = 0
# for line in open(input_file).readlines():
#     i = i+1
#     if i == 1: continue
#     file_name = line.split(",")[0]
#     src_file = os.path.join(input_dir, file_name)
#     print(src_file)
#     #shutils.copy(src_file, "test_img")
#     #os.system("ln -s %s %s" % (src_file, "test_img_50K"))
#     os.system("cp %s %s" % (src_file, "test_img_20K"))
#
#     os.system("rm %s" % (src_file))
#
# '''example:
# train.csv file store tail 50k but drop last 10k lines, save in train_50K.csv
# head -n 1  train.csv > train_50K.csv #store head of csv file
#
# Tail -n 50000  train.csv｜ head -n 40000   >> train_50K.csv
#
# #remove 3000 last line
# tail -n 3000 input.csv >> output.csv
# head -n +3000 input.csv > input.csv.truncated   #head -n +number returns everything but the n last line. ps:dosen't work,
# #just use head -n 955820 instead of it
# mv input.csv.truncated input.csv

file_writer = open("train_new.csv", 'w')
file_writer.write("file_name,SMILES"+"\n")

df = pd.read_csv('/org/temp/anon/data/new_images_5M/train.csv')
data_len = len(df)
print("data length of this group:", data_len)

count = 0
# for idx in tqdm(range(len(filtered_df[filtered_df['group'] == group]))):
for idx in range(data_len):
    smiles = df['SMILES'][idx]  # this is the representation string
    #print(smiles)

    idx = df['file_name'][idx]
    #print(idx)
    #assert len(smiles) <= 100
    if len(smiles) <= 100:
        count += 1
        file_writer.write(idx + "," + smiles + "\n")
        # img_name = str(idx) + ".png"
        # smiles_g = Chem.MolFromSmiles(smiles)
        # try:
        #     # smile_plt is the image so we can directly save it.
        #     smile_plt = Draw.MolToImage(smiles_g, size = (300,300))
        #
        #     img_full_name = os.path.join(img_path, img_name)
        #     file_writer.write(img_name + "," + smiles + "\n")
        #     smile_plt.save(img_full_name)  # save the image in png
        #     assert len(smiles) <= 100
        #     del (smile_plt)
        # except ValueError:
        #     pass
    else:
        pass

    print("count:",count)



input_file = '/org/temp/anon/data/new_images_5M/train.csv'
input_dir = '/org/temp/anon/data/RDkit_SMILES_gray/clear_img/'

output_dir = "test_RDKit"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    pass

i = 0
for line in open(input_file).readlines():
    i = i+1
    if i == 1: continue
    file_name = line.split(",")[0]
    src_file = os.path.join(input_dir, file_name)
    print(src_file)
    #shutils.copy(src_file, "test_img")
    #os.system("ln -s %s %s" % (src_file, "test_img_50K"))
    os.system("cp %s %s" % (src_file, "test_img_20K"))

    os.system("rm %s" % (src_file))

'''example:
train.csv file store tail 50k but drop last 10k lines, save in train_50K.csv
head -n 1  train.csv > train_50K.csv #store head of csv file

Tail -n 50000  train.csv｜ head -n 40000   >> train_50K.csv 

#remove 3000 last line
tail -n 3000 input.csv >> output.csv
head -n +3000 input.csv > input.csv.truncated   #head -n +number returns everything but the n last line. ps:dosen't work, 
#just use head -n 955820 instead of it 
mv input.csv.truncated input.csv
'''






