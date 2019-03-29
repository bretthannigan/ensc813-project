from __future__ import print_function

import networkx as nx
import numpy as np
import argparse
import multiprocessing
from rdkit import Chem
from rdkit.Chem import Draw

NUM_PROCESSES = 8

def get_arguments():
    parser = argparse.ArgumentParser(description='Convert an rdkit Mol to nx graph, preserving chemical attributes')
    parser.add_argument('smiles', type=str, help='The input file containing SMILES strings representing an input molecules.')
    parser.add_argument('nx_pickle', type=str, help='The output file containing sequence of pickled nx graphs')
    parser.add_argument('--num_processes', type=int, default=NUM_PROCESSES, help='The number of concurrent processes to use when converting.')
    return parser.parse_args()
    
def mol_to_nx(mol, is_map_order=False):
    G = nx.Graph()
    idx_to_map_num = {}
    if is_map_order:
        mapped_atoms = {at.GetAtomMapNum() for at in mol.GetAtoms()}.difference({0})
        all_atoms = {i for i in range(1, max(mapped_atoms.union({Chem.Mol.GetNumAtoms(mol)})) + 1)}
        unmapped_atoms = sorted(all_atoms.difference(mapped_atoms))
        i_unmapped_atoms = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() != 0:
                idx_to_map_num[atom.GetIdx()] = atom.GetAtomMapNum() - 1 # AtomMapNum indices are 1-based.
            else:
                idx_to_map_num[atom.GetIdx()] = unmapped_atoms[i_unmapped_atoms] - 1
                i_unmapped_atoms += 1
    else:
        for atom in mol.GetAtoms():
            idx_to_map_num[atom.GetIdx()] = atom.GetIdx()
    for atom in mol.GetAtoms():
        G.add_node(idx_to_map_num[atom.GetIdx()],
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
        G.add_edge(idx_to_map_num[atom.GetIdx()],
                   idx_to_map_num[atom.GetIdx()],
                   bond_type=Chem.rdchem.BondType.OTHER) # Add self-loops.
    for bond in mol.GetBonds():
        G.add_edge(idx_to_map_num[bond.GetBeginAtomIdx()],
                   idx_to_map_num[bond.GetEndAtomIdx()],
                   bond_type=bond.GetBondType())
    return G

def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol

def smiles_to_nx(smiles, validate=False):
    mol = Chem.MolFromSmiles(smiles.strip())
    can_smi = Chem.MolToSmiles(mol)
    G = mol_to_nx(mol)
    if validate:
        mol = nx_to_mol(G)
        new_smi = Chem.MolToSmiles(mol)
        assert new_smi == smiles
    return G

def main():
    args = get_arguments()
    i = open(args.smiles)
    p = multiprocessing.Pool(args.num_processes)
    results = p.map(smiles_to_nx, i.xreadlines())
    o = open(args.nx_pickle, 'w')
    for result in results:
        nx.write_gpickle(result, o)
    o.close()

if __name__ == '__main__':
    main()
