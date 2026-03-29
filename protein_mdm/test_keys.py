import torch
from data.dataset import ProteinStructureDataset
dataset = ProteinStructureDataset('raw_data/1a00.pdb')
sample = dataset[0]
for k in sample.keys():
    print(k, sample[k].shape if hasattr(sample[k], 'shape') else type(sample[k]))
