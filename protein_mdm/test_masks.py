import torch
from data.dataset import ProteinStructureDataset
import numpy as np

dataset = ProteinStructureDataset('raw_data/1a00.pdb')
sample = dataset[0]

levels = sample['fragment_levels'].numpy()
valid_mask = sample['torsion_valid_mask'].numpy()

print(f"Total elements: {len(levels)}")
print(f"Valid mask sum: {valid_mask.sum()}")
print("Valid counts by level:")
for i in range(5):
    valid_in_level = (valid_mask & (levels == i)).sum()
    print(f"  Level {i}: {valid_in_level}")
