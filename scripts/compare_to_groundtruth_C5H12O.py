import ase
from ase import io
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# Ground truth isomers in QM9
with open("../data/C5H12O_ground_truth.smi") as f:
    all_isomers = [line.rstrip() for line in f]
print(all_isomers)


def xyz2smi(path):
    raw_mol = Chem.MolFromXYZFile(path)
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    s = Chem.MolToSmiles(mol, isomericSmiles=False)
    s = Chem.MolToSmiles(Chem.MolFromSmiles(s))
    return s


all_atoms = ase.io.iread("C5H12O_models/structures_20.extxyz")
smi = []
for i in range(20):
    try:
        s = xyz2smi(f"C5H12O_models/structures_20_{i}_relaxed.xyz")
        print("structure", i, "led to smiles", s)
        smi.append(s)
    except:
        print("unable to build smiles from structure", i)

print(smi)
print(set(smi))
print(sum([s in all_isomers for s in set(smi)]))
