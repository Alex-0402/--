import torch
from data.dataset import ProteinStructureDataset
dataset = ProteinStructureDataset('raw_data/1a00.pdb')
sample = dataset[0]
print(torch.unique(sample['fragment_levels']))
