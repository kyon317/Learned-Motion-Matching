# Train Folder - Learned Motion Matching Training Code

This folder contains training code that fully reproduces the original framework. The input/output formats are fully compatible with the original framework and can be used directly with the raylib visualization program.

## File Structure

```
train/
├── models.py                    # Network model definitions (Compressor, Decompressor, Stepper, Projector)
├── models_diffusion.py          # Diffusion-based projector model
├── train_decompressor.py        # Decompressor training script
├── train_stepper.py             # Stepper training script
├── train_projector.py           # Original projector training script (feed-forward network)
├── train_projector_diffusion.py # Diffusion-based projector training script
├── README.md                    # This file
└── README_DIFFUSION.md          # Documentation for the diffusion-based projector
```

## Quick Start

### 1. Training Order

Training must be done in the following order:

```bash
cd train

# Step 1: Train the decompressor (must be done first)
python train_decompressor.py

# Step 2: Train the stepper and projector (can be done in parallel)
python train_stepper.py
python train_projector.py
```

**Alternative:** You can also use the diffusion-based projector instead of the original:

```bash
python train_stepper.py
python train_projector_diffusion.py
```

See `README_DIFFUSION.md` for details about the diffusion-based projector.

### 2. Using the Models

After training, model files will be saved to the `resources/` directory:
- `decompressor.bin`
- `stepper.bin`
- `projector.bin`
- `latent.bin`

These files are fully compatible with the original framework and can be used directly with the raylib visualization program:

```bash
cd ..
make
./controller.exe  # Windows
# or
./controller      # Linux/Mac
```

Enable LMM by checking the **"learned motion matching"** checkbox in the UI.

## Data Paths

- **Input Data**: Read from `../resources/` directory
  - `database.bin`
  - `features.bin`
  
- **Output Models**: Saved to `../resources/` directory
  - `decompressor.bin`
  - `stepper.bin`
  - `projector.bin`
  - `latent.bin`

## Model Architectures

### Decompressor
- **Input**: Feature vector (nfeatures) + Latent variables (nlatent=32)
- **Output**: Bone positions, rotations, velocities, etc.
- **Architecture**: 2-layer fully connected network (512 hidden units)

### Stepper
- **Input**: Feature vector + Latent variables
- **Output**: Feature velocity + Latent variable velocity
- **Architecture**: 3-layer fully connected network (512 hidden units)

### Projector (Original)
- **Input**: Query feature vector
- **Output**: Projected features + Projected latent variables
- **Architecture**: 5-layer fully connected network (512 hidden units)

### Projector (Diffusion-based)
- **Input**: Query feature vector
- **Output**: Projected features + Projected latent variables
- **Architecture**: U-Net style architecture with sinusoidal time embeddings
- **See**: `README_DIFFUSION.md` for detailed architecture information

## Training Parameters

All training scripts use the same parameters as the original framework:

- **Iterations**: 500,000
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: AdamW (amsgrad=True, weight_decay=0.001)
- **Learning Rate Scheduler**: ExponentialLR (gamma=0.99)

## Output Files

During training, the following files are generated:

- **Model Files**: `*.bin` (saved in `resources/` directory)
- **Visualization Images**: `*.png` (training progress visualization)
- **Test Animations**: `*.bvh` (for validation)
- **TensorBoard Logs**: `resources/runs/` directory

To view TensorBoard:

```bash
tensorboard --logdir=../resources/runs
```

## Compatibility

- ✅ Input/output formats are fully compatible with the original framework
- ✅ Model file formats are fully compatible with the original framework
- ✅ Can be used directly with the raylib visualization program
- ✅ Uses the same training parameters and loss functions

## Troubleshooting

### Import Errors

If you encounter `ImportError`, ensure:
1. The `resources/` directory contains `quat.py`, `txform.py`, `tquat.py`, `bvh.py`
2. Python path is set correctly

### File Not Found

Ensure the `resources/` directory contains:
- `database.bin`
- `features.bin` (if not present, running the visualization program will generate it automatically)

### Training Fails

Check:
1. Data files are complete
2. Sufficient memory is available
3. PyTorch is correctly installed

## References

- Original training code: `../resources/train_*.py` (if present)
- Visualization program: `../controller.cpp`
- Main README: `../README.md`
- Diffusion projector documentation: `README_DIFFUSION.md`
