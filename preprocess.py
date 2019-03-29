import sys
import csv

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw

import ReactionGraph

NUM_RXNS_TO_READ = 10

with open("data/1976_Sep2016_USPTOgrants_smiles.rsmi", newline='') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    next(csv_reader)
    line_count = 0
    reactions = []
    for row in csv_reader:
        reactions.append(Chem.ReactionFromSmarts(row[0].split(None, 1)[0]))
        line_count += 1
        if line_count == NUM_RXNS_TO_READ:
            break

num_reactant_atoms = np.zeros(len(reactions), dtype=np.int32)
num_product_atoms = np.zeros(len(reactions), dtype=np.int32)
for i, rxn in enumerate(reactions):
    reac = Chem.ChemicalReaction.GetReactants(rxn)
    prod = Chem.ChemicalReaction.GetProducts(rxn)
    for r in reac:
        num_reactant_atoms[i] += r.GetNumAtoms()
    for p in prod:
        num_product_atoms[i] += p.GetNumAtoms()
Chem.SanitizeRxn(reactions[0])
test = ReactionGraph.ReactionGraph.from_ChemicalReaction(reactions[8])
test.num_atoms = 100
y = test.A_reac
test.f_reac
mol = Chem.ChemicalReaction.GetReactants(reactions[7])[0]
Draw.MolToImage(mol).show()
Chem.SanitizeMol(mol)
Draw.MolToImage(mol).show()
m2 = Chem.MolFromSmiles('[CH3:1][C:2]1[N:3]=[CH:4][C:5]2=[CH:6][CH:7]=[CH:8][C:9](=[C:10]2[CH:11]=1)[N:12](=[O:13])[OH:14]')
x = 5