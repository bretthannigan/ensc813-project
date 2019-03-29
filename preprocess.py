import sys
import csv

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw

import ReactionGraph

NUM_RXNS_TO_READ = 10000

with open("data/1976_Sep2016_USPTOgrants_smiles.rsmi", newline='') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    next(csv_reader)
    reaction = []
    try_count = 0
    success_count = 0
    fail_count = 0
    error = False
    for row in csv_reader:
        fail_count += error
        error = False
        try_count += 1
        row = row[0].split(None, 1)[0]
        reac, _, prod = row.split(">")
        reac = reac.split(".")
        reac = [Chem.MolFromSmiles(r) for r in reac]
        try:
            for r in reac:
                Chem.SanitizeMol(r)
        except:
            error = True
            print("\tSkipping reaction on line {}, error in reactant SMILES.".format(try_count + 1))
            continue
        prod = prod.split(".")
        prod = [Chem.MolFromSmiles(p) for p in prod]
        try:
            for p in prod:
                Chem.SanitizeMol(p)
        except:
            error = True
            print("\tSkipping reaction on line {}, error in products SMILES".format(try_count + 1))
            continue
        reaction.append(ReactionGraph.ReactionGraph.from_rdMol(reac, prod))
        success_count += 1
        if success_count >= NUM_RXNS_TO_READ:
            break
    print("Attempted import of {} reactions with {} successes and {} errors.".format(try_count, success_count, fail_count))