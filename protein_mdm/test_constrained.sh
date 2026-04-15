#!/bin/bash
MODEL="checkpoints_20000_0328/best_model.pt"
PDB="raw_data/1a00.pdb"

echo "=== RUNNING ORIGINAL ==="
python inference.py --model_path "$MODEL" --pdb_path "$PDB" --output_dir test_out_orig --num_iterations 12 --strategy adaptive > orig_run.log 2>&1

echo "=== RUNNING CONSTRAINED ==="
python inference_constrained.py --model_path "$MODEL" --pdb_path "$PDB" --output_dir test_out_constrained --num_iterations 12 --strategy adaptive > constrained_run.log 2>&1

echo "Done! Check test_out_orig and test_out_constrained"
