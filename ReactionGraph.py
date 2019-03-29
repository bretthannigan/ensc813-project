import scipy.linalg
import scipy.sparse as sp
import numpy as np
import networkx as nx
from rdkit.Chem import AllChem as Chem
import convert_rdkit_to_networkx
from sklearn.preprocessing import OneHotEncoder

class ReactionGraph:

    # Define labels, object type labels are converted to integers.
    _atomic_num_categories = np.atleast_2d(np.asarray([1, 6, 7, 8, 9, 15, 16, 17, 35]))
    _chirality_categories = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW] # R- or S- chirality
    _chirality_categories = np.atleast_2d(np.asarray([int(i) for i in _chirality_categories], dtype=np.int32))
    _hybridization_categories = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]
    _hybridization_categories = np.atleast_2d(np.asarray([int(i) for i in _hybridization_categories], dtype=np.int32))
    # Intstantiate one hot encoders, must fit to labels to ensure all ReactionGraph objects have same one hot encoding format.
    _atomic_num_encoder = OneHotEncoder(categories=_atomic_num_categories, handle_unknown='ignore')
    _atomic_num_encoder.fit(_atomic_num_categories.T)
    _chirality_encoder = OneHotEncoder(categories=_chirality_categories, handle_unknown='ignore')
    _chirality_encoder.fit(_chirality_categories.T)
    _hybridization_encoder = OneHotEncoder(categories=_hybridization_categories, handle_unknown='ignore')
    _hybridization_encoder.fit(_hybridization_categories.T)

    def __init__(self, reac=[], prod=[], feat=[], natoms=None):
        self.reactant = reac
        self.product = prod
        self._num_atoms = natoms
    
    @classmethod
    def from_ChemicalReaction(cls, rxn):
        reactant = []
        product = []
        for mol in Chem.ChemicalReaction.GetReactants(rxn):
            reactant.append(convert_rdkit_to_networkx.mol_to_nx(mol))
        for mol in Chem.ChemicalReaction.GetProducts(rxn):
            product.append(convert_rdkit_to_networkx.mol_to_nx(mol))
        return cls(reactant, product)

    @classmethod
    def from_rdMol(cls, reac, prod):
        reactant = []
        product = []
        for mol in reac:
            reactant.append(convert_rdkit_to_networkx.mol_to_nx(mol))
        for mol in prod:
            product.append(convert_rdkit_to_networkx.mol_to_nx(mol))
        return cls(reactant, product)

    @property
    def A_reac(self):
        A = [nx.adjacency_matrix(i) for i in self.reactant]
        A = scipy.sparse.block_diag(A, format='csr')
        if np.any(self.num_atoms - np.asarray(A.shape)):
            A.resize(self.num_atoms, self.num_atoms)
        return A

    @property
    def A_prod(self):
        A = [nx.adjacency_matrix(i) for i in self.product]
        A = scipy.sparse.block_diag(A, format='csr')
        if np.any(self.num_atoms - np.asarray(A.shape)):
            A.resize(self.num_atoms, self.num_atoms)
        return A

    @property
    def f_reac(self):
        n = self.num_atoms
        atomic_num = np.zeros((n, 1))
        chirality = np.zeros((n, 1))
        hybridization = np.zeros((n, 1))
        formal_charge = np.zeros((n, 1))
        is_aromatic = np.zeros((n, 1))
        num_explicit_h = np.zeros((n, 1))
        offset = 0
        for mol in self.reactant:
            for j in range(len(mol.nodes())):
                atomic_num[offset+j] = mol.node[j]['atomic_num']
                chirality[offset+j] = mol.node[j]['chiral_tag']
                hybridization[offset+j] = mol.node[j]['hybridization']
                formal_charge[offset+j] = mol.node[j]['formal_charge']
                is_aromatic[offset+j] = mol.node[j]['is_aromatic']
                num_explicit_h[offset+j] = mol.node[j]['num_explicit_hs']
            offset += nx.number_of_nodes(mol)
        atomic_num = self._atomic_num_encoder.transform(atomic_num)
        chirality = self._chirality_encoder.transform(chirality)
        hybridization = self._hybridization_encoder.transform(hybridization)
        return sp.hstack((atomic_num, chirality, hybridization, sp.csr_matrix(formal_charge), sp.csr_matrix(is_aromatic), sp.csr_matrix(num_explicit_h)))

    @property
    def num_atoms(self):
        if self._num_atoms is None:
            if not self.reactant:
                return 0
            else:
                return np.sum([nx.number_of_nodes(i) for i in self.reactant])
        else:
            return self._num_atoms

    @num_atoms.setter
    def num_atoms(self, na):
        self._num_atoms = na