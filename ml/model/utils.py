import os
import csv
import json
import argparse
#from ..model.src.config import data_dir
import pubchempy as pcp
import pandas as pd
from rdkit import Chem
from mol2chemfigPy3 import mol2chemfig

def _csv_writer(file_name, write_data):
    f = open(file_name, 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(write_data)
    f.close()

# from ..model.src.config import data_dir
# def logger(log_data):
#     _csv_writer(data_dir + '/log.csv', log_data)
def logger(log_data):
    _csv_writer('log.csv', log_data)

def make_directory(path):
    try:
        os.mkdir(path)
        print(path + ' is generated!')
    except OSError:
        pass

def load_reversed_token_map(path):
    """Gets the path of the reversed token map json"""
    with open(path, 'r') as j:
        reversed_token_map = json.load(j)
    return reversed_token_map

def decode_predicted_sequences(predicted_sequence_list,reversed_token_map):
    """
    :param predicted_sequence_list: List of sequences in predicted form ex) [27,1,2,5]
    :param reveresed_token_map: Dictionary mapping of reversed token map
    :return: predicted_sequence_str:
    """
    predicted_sequence_str = ""
    for e in predicted_sequence_list:
        e = str(e)
        if reversed_token_map[e]=='<unk>':
            continue
        elif reversed_token_map[e] in {'<end>','<pad>'}:
            break
        else:
            predicted_sequence_str+=reversed_token_map[e]

    return predicted_sequence_str


async def async_decode_predicted_sequences(predicted_sequence_list, reversed_token_map):
    """
    :param predicted_sequence_list: List of sequences in predicted form ex) [27,1,2,5]
    :param reveresed_token_map: Dictionary mapping of reversed token map
    :return: predicted_sequence_str:
    """
    predicted_sequence_str = ""
    for e in predicted_sequence_list:
        e = str(e)
        if reversed_token_map[e] == '<unk>':
            continue
        elif reversed_token_map[e] in {'<end>', '<pad>'}:
            break
        else:
            predicted_sequence_str += reversed_token_map[e]

    return predicted_sequence_str

def smiles_name_print():
    print('  ______   __    __   __   __       ______   ______    ')
    print(' /\  ___\ /\ "-./  \ /\ \ /\ \     /\  ___\ /\  ___\   ')
    print(' \ \___  \\\\ \ \-./\ \\\\ \ \\\\ \ \____\ \  __\ \ \___  \  ')
    print('  \/\_____\\\\ \_\ \ \_\\\\ \_\\\\ \_____\\\\ \_____\\\\/\_____\ ')
    print('   \/_____/ \/_/  \/_/ \/_/ \/_____/ \/_____/ \/_____/ ')


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert_smiles(smiles):
    try:
        compounds = pcp.get_compounds(smiles, namespace='smiles')
        c = compounds[0]
        input_smiles = smiles
        sum_formula = c.molecular_formula
        iupac_name = c.iupac_name
        isomeric_smiles = c.isomeric_smiles
        #canonical_smiles = match.canonical_smiles #Canonical SMILES, with no stereochemistry information.
        inchi = c.inchi
        inchikey = c.inchikey
        synonyms = c.synonyms
        #Because PubChem only support isomeric SMILES, So we generate canonical SMILES from RDKit.
        mol_inchi = Chem.inchi.MolFromInchi(inchi)
        smiles_cano = Chem.MolToSmiles(mol_inchi, canonical=True, isomericSmiles=False)

        #Molecular Weight:
        weights = c.molecular_weight
        #chem = mol2chem(inchi)
        elements = c.elements
        elem_num = elements_num(elements)
        bonds_num = c.bonds
        atoms_num = c.atoms

        descrip = description(elem_num,bonds_num,atoms_num,weights)
        print('CID in PubChem:',compounds)
        #print('Input SMILES:',input_smiles)
        print('IUPAC Name:',iupac_name)
        print('Sum Formula:',sum_formula)
        print('Isomeric SMILES:',isomeric_smiles)
        print('Other Canonical SMILES:',smiles_cano)
        print('InChI:', inchi)
        print('InChIKeys:', inchikey)
        print('Chemfig:')
        #print(chem)
        mol2chem(inchi)
        print(descrip)

        print('Synonyms:',synonyms)
        #print('='*200)

    except:
        print("Invalid SMILES: This SMILES string doesn't exist in PubChem library.")



def elements_num(elements):
    ls = {}
    n = 0
    count = 1
    for idx in range(len(elements)-1):
        #if idx < len(elements)-1:
        ele = elements[idx]
        #print(ele)

        if elements[idx+1] != elements[idx] or idx == len(elements) - 2:
            ls.update({ele: count})
            count = 1
        else:
            count += 1
    #the number of last element should be added 1.
    last_ele = list(ls.keys())[-1]
    last_value = list(ls.values())[-1]
    last_value = last_value + 1
    ls[last_ele] = last_value
    #print(list(ls.keys())[-1])
    #print(last_value)

    return ls


def description(elements_num,bonds_num,atoms_num,weights):
    key = list(elements_num.keys())
    key_string = ','.join([str(elem) for elem in key]) #convert list into string
    values = list(elements_num.values())
    values_string = ','.join([str(elem) for elem in values])
    bonds_num = len(bonds_num)
    atoms_num = len(atoms_num)
    elem_num = len(elements_num)
    weights = weights
    des = ("This chemical structure contains {1} atoms and {2} bonds between them. There are {3} elements {4}, the corresponding number for each element is {5}. Its weights is {0} in total.".format(weights, atoms_num,bonds_num, elem_num, key_string, values_string))
    return des

def mol2chem(inchi):

    return mol2chemfig(inchi)

