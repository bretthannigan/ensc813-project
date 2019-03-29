import scipy.linalg
import scipy.sparse as sp
import numpy as np
import networkx as nx
from rdkit.Chem import AllChem as Chem
import convert_rdkit_to_networkx
from sklearn.preprocessing import OneHotEncoder

class ReactionSideGraph:

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

    def __init__(self, spec=[], natoms=None):
        self.species = spec
        self._num_atoms = natoms
    
    @classmethod
    def from_ChemicalReaction(cls, rxn, side='reactants', natoms=None):
        species = []
        if side == 'reactants':
            for mol in Chem.ChemicalReaction.GetReactants(rxn):
                species.append(convert_rdkit_to_networkx.mol_to_nx(mol))
        elif side == 'products':
            for mol in Chem.ChemicalReaction.GetProducts(rxn):
                species.append(convert_rdkit_to_networkx.mol_to_nx(mol))
        return cls(species, natoms)

    @classmethod
    def from_rdMol(cls, spec=[], natoms=None):
        species = []
        for mol in spec:
            species.append(convert_rdkit_to_networkx.mol_to_nx(mol, is_map_order=True))
        return cls(species, natoms)

    def get_adjacency(self, normalize=False):
        A = [nx.adjacency_matrix(i, nodelist=range(max(list(i.nodes)) + 1)) for i in self.species]
        A = scipy.sparse.block_diag(A, format='csr')
        if np.any(self.num_atoms - np.asarray(A.shape)):
            A.resize(self.num_atoms, self.num_atoms)
        if normalize:
            # Normalize using rule from: T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” pp. 1–14, 2016.
            #   D^(-1/2)*A*D^(-1/2)
            D_inv = self.get_degree(inv=True)
            A = sp.dia_matrix.sqrt(D_inv).dot(A).dot(sp.dia_matrix.sqrt(D_inv)).tocsr()
        return A

    def get_degree(self, inv=False):
        # Note: the networx implementation of degree() counts self-loops as 2 degrees whereas the gcn implementation counts them as one.
        degree = []
        for spec in self.species:
            degree_view = nx.degree(spec)
            degree_vect = np.zeros(max(list(spec.nodes)) + 1)
            for i, d in degree_view:
                degree_vect[i] = d
            degree.append(degree_vect)
        degree_diag = np.concatenate(degree)
        if inv:
            degree_diag = 1./degree_diag
            degree_diag[np.isinf(degree_diag)] = 0
        D = sp.diags(degree_diag)
        if np.any(self.num_atoms - np.asarray(D.shape)):
            D.resize(self.num_atoms, self.num_atoms)
        return D

    @staticmethod
    def _degree_inv(A):
        d = np.array(A.sum(1)).flatten()
        d_inv = 1./d
        d_inv[np.isinf(d_inv)] = 0.
        return sp.diags(d_inv)

    def get_features(self):
        n = self.num_atoms
        atomic_num = np.zeros((n, 1))
        chirality = np.zeros((n, 1))
        hybridization = np.zeros((n, 1))
        formal_charge = np.zeros((n, 1))
        is_aromatic = np.zeros((n, 1))
        num_explicit_h = np.zeros((n, 1))
        offset = 0
        for spec in self.species:
            for j in list(spec.nodes()):
                atomic_num[offset+j] = spec.node[j]['atomic_num']
                chirality[offset+j] = spec.node[j]['chiral_tag']
                hybridization[offset+j] = spec.node[j]['hybridization']
                formal_charge[offset+j] = spec.node[j]['formal_charge']
                is_aromatic[offset+j] = spec.node[j]['is_aromatic']
                num_explicit_h[offset+j] = spec.node[j]['num_explicit_hs']
            offset += nx.number_of_nodes(spec)
        atomic_num = self._atomic_num_encoder.transform(atomic_num)
        chirality = self._chirality_encoder.transform(chirality)
        hybridization = self._hybridization_encoder.transform(hybridization)
        return sp.hstack((atomic_num, chirality, hybridization, sp.csr_matrix(formal_charge), sp.csr_matrix(is_aromatic), sp.csr_matrix(num_explicit_h)))

    @property
    def num_atoms(self):
        if self._num_atoms is None:
            if not self.species:
                return 0
            else:
                return np.sum([nx.number_of_nodes(i) for i in self.species])
        else:
            return self._num_atoms

    @num_atoms.setter
    def num_atoms(self, na):
        self._num_atoms = na