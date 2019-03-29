import sys
import csv

import numpy as np
import pickle
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
import scipy.sparse as sp

import ReactionGraph as rg

DATA_FILE = "data/1976_Sep2016_USPTOgrants_smiles.rsmi"
NUM_RXNS_TO_READ = 1000
MAX_NUM_ATOMS = 128

with open(DATA_FILE, newline='') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    next(csv_reader)
    reactant = []
    product = []
    try_count = 0
    success_count = 0
    fail_count = 0
    is_error_on_iter = False
    for row in csv_reader:
        fail_count += is_error_on_iter
        is_error_on_iter = False
        try_count += 1
        row = row[0].split(None, 1)[0]
        # Reaction SMILES format: reactant_1.reactant_2>reagent>product_1.product_2
        reac, _, prod = row.split(">") # Not interested in reagents/catalysts.
        reac = Chem.MolFromSmiles(reac)
        try:
            Chem.SanitizeMol(reac)
            num_reactant_atoms = Chem.Mol.GetNumAtoms(reac)
        except:
            is_error_on_iter = True
            print("\tSkipping reaction on line {}, error in reactant SMILES.".format(try_count + 1))
            continue
        prod = Chem.MolFromSmiles(prod)
        try:
            Chem.SanitizeMol(prod)
            num_product_atoms = Chem.Mol.GetNumAtoms(prod)
        except:
            is_error_on_iter = True
            print("\tSkipping reaction on line {}, error in product SMILES".format(try_count + 1))
            continue
        if num_reactant_atoms>MAX_NUM_ATOMS or num_product_atoms>MAX_NUM_ATOMS:
            is_error_on_iter = True
            print("\tSkipping reaction on line {}, has greater than {} atoms.".format(try_count + 1, MAX_NUM_ATOMS))
            continue
        reactant.append(rg.ReactionSideGraph.from_rdMol([reac], MAX_NUM_ATOMS))
        product.append(rg.ReactionSideGraph.from_rdMol([prod], MAX_NUM_ATOMS))
        success_count += 1
        if success_count >= NUM_RXNS_TO_READ:
            break
    print("Attempted import of {} reactions with {} successes and {} skips.".format(try_count, success_count, fail_count))

with open("LoweUSPTOGrants_1976-2016_{}Atoms_{}Reactions.pickle".format(MAX_NUM_ATOMS, NUM_RXNS_TO_READ), 'wb') as f:
    pickle.dump(reactant, f)
    pickle.dump(product, f)
# with open("LoweUSPTOGrants_1976-2016_128Atoms_1000Reactions.pickle", 'rb') as f:
#     e = pickle.load(f)
#     g = pickle.load(f)

A = [r.get_adjacency(normalize=True) for r in reactant]
X = [r.get_features() for r in reactant]
y = [p.get_adjacency(normalize=True) for p in product]
X_y = [p.get_features() for p in product]
x = 5