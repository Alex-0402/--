import math
import numpy as np

# Variables
vocab_size_fragment = 12 + 4 # 12 elements + PAD, MASK, BOS, EOS (approx 16-17 types)
vocab_size_torsion = 72

# Random guess loss
random_frag_loss = math.log(vocab_size_fragment)
random_torsion_loss = math.log(vocab_size_torsion)

print(f"Random Fragment Loss: {random_frag_loss:.4f}")
print(f"Random Torsion Loss: {random_torsion_loss:.4f}")

# Target heuristic bounds
print(f"Target Frag (90% conf): {-math.log(0.9):.4f}")
print(f"Target Frag (80% conf): {-math.log(0.8):.4f}")
print(f"Target Torsion (Within +/- 1 bin / 15 deg): {-math.log(3/72):.4f}")

