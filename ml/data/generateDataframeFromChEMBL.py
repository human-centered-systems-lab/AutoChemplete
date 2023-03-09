'''
Download ChEMBL 30.csv fro ChEMBL,which is [rovided under a Creative Commons Attribution-ShareAlike 3.0 Unported license
Last Update on 2022-02-22T00:00:00  |  Release notes
2.2M Compounds

Download csv file , extract corresponding SMILES and "ChEMBL ID".
'''

import pandas as pd

'''Extract ChEMBL ID and SMILES'''
f = open('ChEMBL30.csv') # len(list(f)): 2157380
#save length smaller than 75
smiles = []
chem_id = []

#save length between 75 and 100
smiles_100 = []
chem_id_100 = []

count = 0
count_100 = 0
#2157370
next(f) #skip header
for i in range(2157370):
    line = f.readline()
    #print(line)
    l = line.split(';')
    #print(len(l))
    if len(l) == 32:
        #print("succeed")
        #read ChEMBL ID
        id = l[0]
        id = id[:-1]
        id = id[1:]
        #print(id)
        #read Canonical SMILES sting
        s = l[-2]
        s = s[:-1]
        s = s[1:]
        #print(len(s))
        #print(s)
        if len(s) <= 75:
            count += 1
            chem_id.append(id)
            smiles.append(s)
        if len(s) > 75 and len(s) <= 100:
            count_100 += 1
            chem_id_100.append(id)
            smiles_100.append(s)
        else:
            pass
    else:
        print("Error! sequence not found, cloumns is {0}.".format(len(l)))


    # if i % 10000 == 0:
    #     print("Number of completion:{0}".format(i))

print("Number of length <=75 is {0}".format(count))
print("Number of length between 75 and 100 is {0}".format(count_100))

#save into dataframe
d = {'ChEMBL ID': chem_id, 'SMILES': smiles}
df = pd.DataFrame(d)

d_100 = {'ChEMBL ID': chem_id_100, 'SMILES': smiles_100}
df_100 = pd.DataFrame(d_100)
print(df)
print(df_100)
#print(len(df['SMILES'][3]))

df.to_csv('chemID+SMILES.csv')
df_100.to_csv('chemID+SMILES_75_100.csv')
'''
Number of length <=75 is 1882701
Number of length between 75 and 100 is 146131
             ChEMBL ID                                             SMILES
0        CHEMBL3425773                  N#Cc1ccc(-c2coc3cc(O)ccc3c2=O)cc1
1        CHEMBL1867860  COc1cccc(-n2c(=O)c3c(C)c(C)sc3n(CC(=O)Nc3ccccc...
...                ...                                                ...
1882699   CHEMBL542534             Br.CCCN(CCC)[C@H]1CCc2c(O)cccc2[C@H]1C
1882700  CHEMBL3249867  CC(C)O.Cl.Oc1ccc2c(c1)C1(c3ccccc3)CCN(CC3CC3)C...
[1882701 rows x 2 columns]

            ChEMBL ID                                             SMILES
0       CHEMBL1998753  COC(=O)[C@@H]1[C@H]2[C@H](OC(=O)c3ccccc3)CCN2O...
1       CHEMBL1825079  CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@H]3[C@H](CC[...
...               ...                                                ...
146130    CHEMBL27618  COc1cc([C@@H]2c3cc4c(cc3[C@@H](NCCN3CCCCC3)[C@...
[146131 rows x 2 columns]
'''

