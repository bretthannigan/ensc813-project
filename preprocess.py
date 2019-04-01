import sys
import csv

import numpy as np
import pickle
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
import scipy.sparse as sp

import ReactionGraph as rg

def import_patent_dataset(data_file="data/1976_Sep2016_USPTOgrants_smiles.rsmi", num_reactions=np.Inf, max_num_atoms=128):

    with open(data_file, newline='') as f:
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
            if num_reactant_atoms>max_num_atoms or num_product_atoms>max_num_atoms:
                is_error_on_iter = True
                print("\tSkipping reaction on line {}, has greater than {} atoms.".format(try_count + 1, max_num_atoms))
                continue
            reactant.append(rg.ReactionSideGraph.from_rdMol([reac], max_num_atoms))
            product.append(rg.ReactionSideGraph.from_rdMol([prod], max_num_atoms))
            success_count += 1
            if success_count >= num_reactions:
                break
        print("Attempted import of {} reactions with {} successes and {} skips.".format(try_count, success_count, fail_count))

    with open("/data/LoweUSPTOGrants_1976-2016_{}Atoms_{}Reactions.pickle".format(max_num_atoms, success_count), 'wb') as f:
        pickle.dump(reactant, f)
        pickle.dump(product, f)

def import_logp_dataset(data_file="data/logP_dataset.csv", num_compounds=100000, max_num_atoms=32):
    with open(data_file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        compound = []
        log_p = []
        try_count = 0
        success_count = 0
        fail_count = 0
        for row in csv_reader:
            try_count += 1
            new_compound = Chem.MolFromSmiles(row[0])
            try:
                Chem.SanitizeMol(new_compound)
            except:
                fail_count += 1
                print("\tSkipping compound on line {}, error in SMILES".format(try_count))
                continue
            compound.append(rg.ReactionSideGraph.from_rdMol([new_compound], max_num_atoms))
            log_p.append(float(row[1]))
            success_count += 1
    print("Attempted import of {} reactions with {} successes and {} skips.".format(try_count, success_count, fail_count))

    with open("KagglelogP.pickle", 'wb') as f:
        pickle.dump(compound, f)
        pickle.dump(log_p, f)

import_logp_dataset()