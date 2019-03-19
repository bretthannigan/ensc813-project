import numpy as np
import csv
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

with open("data/1976_Sep2016_USPTOgrants_smiles.rsmi", newline='') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    next(csv_reader)
    for row in csv_reader:
        print('a')
        rxn = AllChem.ReactionFromSmarts(row[0].split(None, 1)[0])
        