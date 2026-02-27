# Copilot Instructions for `protein_mdm`

This document provides essential guidance for AI coding agents working on the `protein_mdm` codebase. It highlights the architecture, workflows, and conventions specific to this project.

---

## Project Overview

`protein_mdm` is a Python-based project for protein modeling and analysis. The codebase is modular, with distinct components for data processing, model architecture, and utilities. Key innovations include:

- **Fragment Vocabulary Mapping**: Translating amino acids into rigid chemical fragments.
- **Torsion Angle Analysis**: Extracting and discretizing torsion angles from protein structures.

---

## Codebase Structure

### Key Directories

- **`data/`**: Handles data processing.
  - `vocabulary.py`: Core file for fragment vocabulary mapping.
  - `geometry.py`: Tools for torsion angle calculations.
  - `dataset.py`: PDB dataset loader.
- **`models/`**: Defines the model architecture.
  - `encoder.py`: Backbone encoder (torch-geometric).
  - `decoder.py`: Predicts fragments and torsion angles.
- **`utils/`**: Utility functions.
  - `protein_utils.py`: Biopython helpers.
- **`scripts/`**: Scripts for preprocessing and benchmarking.
- **`checkpoints/`**: Stores model checkpoints.
- **`raw_data/`**: Contains raw PDB files and metadata.

### Key Files

- `main.py`: Entry point for the project.
- `test_all.py`: Comprehensive testing script.
- `requirements.txt`: Lists dependencies.
- `setup_env.sh`: Environment setup script.

---

## Developer Workflows

### Environment Setup

1. Install dependencies:
   ```bash
   bash setup_env.sh
   ```
2. (Optional) Use Conda:
   ```bash
   conda env create -f environment.yml
   conda activate protein_mdm
   ```

### Running Tests

- Run all tests:
  ```bash
  python test_all.py
  ```
- Test specific modules (e.g., `test_models.py`):
  ```bash
  python -m unittest test_models.py
  ```

### Training and Inference

- Train the model:
  ```bash
  python train.py --config configs/train_config.yaml
  ```
- Run inference:
  ```bash
  python inference.py --model_path checkpoints/best_model.pt --pdb_path raw_data/sample.pdb --output_path predictions.pdb
  ```

---

## Project-Specific Conventions

- **Fragment Vocabulary**: Use `data/vocabulary.py` for amino acid-to-fragment mappings.
- **Torsion Angles**: Use `data/geometry.py` for dihedral angle calculations.
- **Testing**: Place all test files in the root directory, prefixed with `test_`.
- **Checkpoints**: Store model checkpoints in `checkpoints/` with clear naming (e.g., `checkpoint_epoch_10.pt`).

---

## Integration Points

- **Dependencies**: Core libraries include `torch`, `torch-geometric`, and `biopython`.
- **Data Flow**: Raw PDB files are processed into fragments and angles, then fed into the model.
- **Scripts**: Use `scripts/` for preprocessing and benchmarking tasks.

---

## Examples

### Adding a New Test

1. Create a file `test_new_feature.py` in the root directory.
2. Use `unittest` framework:
   ```python
   import unittest

   class TestNewFeature(unittest.TestCase):
       def test_case(self):
           self.assertEqual(1 + 1, 2)

   if __name__ == "__main__":
       unittest.main()
   ```
3. Run the test:
   ```bash
   python test_new_feature.py
   ```

### Adding a New Script

1. Place the script in `scripts/`.
2. Follow the naming convention: `verb_noun.py` (e.g., `preprocess_data.py`).
3. Document usage in the script header.

---

For further details, refer to `PROJECT_STRUCTURE.md` and `README.md`. Feedback and updates to this document are welcome.