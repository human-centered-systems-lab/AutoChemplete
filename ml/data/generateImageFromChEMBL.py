
'''Generate images from ChEMBL web service

reference:
https://github.com/chembl/GLaDOS-docs/blob/master/web-services/chembl-data-web-services.md
https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services#molecule-images
'''


import pandas as pd
import os
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import requests
import cairosvg
from cairosvg import svg2png

#remove the nan value
# df = pd.read_csv('chemID+SMILES.csv')
# print(len(df)) #1882701
# df = df[df['SMILES'].notna()]
# print(len(df)) #1861548
# df.to_csv('chemID+SMILES_notnan.csv')

# df = pd.read_csv('chemID+SMILES_75_100.csv')
# print(len(df)) #146131
# df = df[df['SMILES'].notna()]
# print(len(df)) #146131
# df.to_csv('chemID+SMILES_75_100_notnan.csv')

df = pd.read_csv('chemID+SMILES_notnan.csv')
path = '/org/temp/anon/data/ChEMBL/imgFromChEMBL' # Saving new data
if not os.path.exists(path):
    os.mkdir(path)
else:
    pass

img_path = path + '/train'
#img_path = './train'

if not os.path.exists(img_path):
    os.mkdir(img_path)
else:
    pass

file_writer = open(path + "/train.csv", 'w')
file_writer.write("file_name,SMILES"+"\n")



img_num = 0
print(len(df))
for _, row in df.head(1761548).iterrows():
    #print(_)
    #print(row)
    id = row['ChEMBL ID']
    #print(id)
    smiles = row['SMILES']

    #print(smiles)
    img_name = str(img_num) + '.png'
    #print("img_name:", img_name)
    img_num += 1
    img_full_name = os.path.join(img_path, img_name)
    #svg_full_name = os.path.join(img_path, id)
    url_head = 'https://www.ebi.ac.uk/chembl/api/data/image/'
    url = url_head + id
    print(url)
    try:
        #access web service by ChEMBL
        #request image from web service
        img_svg = requests.get(url,timeout=10)
        #open(svg_full_name, 'wb').write(img_svg.content)
    except ValueError:
        print("Image" + id + " couldn't be requested in 10s.")
    else:
        svg = img_svg.content
        #print(svg)
        svg2png(bytestring=svg, write_to=img_full_name)
        file_writer.write(img_name + "," + smiles + "\n")
        del (img_svg)
        del (svg)


#draw with the library RDKit
        # smiles_g = Chem.MolFromSmiles(smiles)
        # # smile_plt is the image so we can directly save it.
        # smile_plt = Draw.MolToImage(smiles_g, size = (300,300))
        # smile_plt.save(img_full_name)
        # file_writer.write(img_name + "," + smiles + "\n")
        # assert len(smiles) <= 100
        # del (smile_plt)

        # except ValueError:
        #     pass



    if img_num % 1000 == 0 :
        print('finish index : {0}'.format(img_num))
del(df)
file_writer.close()