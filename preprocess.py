import sys
print(sys.path)
import csv

import numpy as np
#import rdkit
from rdkit.Chem import AllChem as Chem

NUM_RXNS_TO_READ = 10000

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

x = 5