
import pandas as pd

from src.config import generate_submission_dir, sample_submission_labels_dir
from rdkit import Chem, DataStructs

#python /org/temp/anon/data/model/tanimotoSmilarity.py
predict_file = pd.read_csv(generate_submission_dir)
labels_file = pd.read_csv(sample_submission_labels_dir)
count = 0
sum_Tan = 0

#print(generate_submission_dir)
for _, row in predict_file.iterrows(): #Iterate over DataFrame rows as (index, Series) pairs.
    idx = row['file_name']
    smiles_pred = row['SMILES']
    smiles_label = labels_file.loc[_]['SMILES']
    label_idx = labels_file.loc[_]['file_name']
    print("_", _)
    print('pred_idx', idx)
    print('smiles_pred:', smiles_pred)
    print("smiles_label:", smiles_label)
    print('label_idx:', label_idx)
    try:
        ref_pred = Chem.MolFromSmiles(smiles_pred)
        fp_pred = Chem.RDKFingerprint(ref_pred)
    except:
        print('Invalid SMILES:', smiles_pred)
        smiles_pred = 'C'
        ref_pred = Chem.MolFromSmiles(smiles_pred)
        fp_pred = Chem.RDKFingerprint(ref_pred)


    ref_label = Chem.MolFromSmiles(smiles_label)
    fp_label = Chem.RDKFingerprint(ref_label)

    Tan = DataStructs.TanimotoSimilarity(fp_pred,fp_label)
    print(Tan)
    sum_Tan = sum_Tan + Tan
    count += 1

average = sum_Tan/count
print("average:", average)
