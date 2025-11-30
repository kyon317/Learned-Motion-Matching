# Motion Matching & Code vs Data Driven Displacement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language](https://img.shields.io/badge/language-C%2B%2B-orange.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](https://github.com)
[![raylib](https://img.shields.io/badge/raylib-5.0-green.svg)](https://www.raylib.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Deep%20Learning-red.svg)](https://github.com)
[![Web](https://img.shields.io/badge/web-Emscripten%20%7C%20WASM-purple.svg)](https://emscripten.org/)
[![Research](https://img.shields.io/badge/research-Motion%20Matching-brightgreen.svg)](https://github.com)

This repository contains my implementation of Learned Motion Matching, based on the original work described in [this article](https://theorangeduck.com/page/code-vs-data-driven-displacement) and [this paper](https://theorangeduck.com/page/learned-motion-matching). I also implemented a diffusion based network to replace the original Projector network, you can check details in the Training section.

## 🎮 Live Demo

[**Try the interactive web demo →**](https://kyon317.github.io/Motion-Matching/controller.html)

The web demo runs entirely in your browser using WebAssembly. All pre-trained models are included, so you can interact with the motion matching system without any installation.

This implementation includes custom training scripts in the `train/` folder, including an alternative diffusion-based projector network, while maintaining full compatibility with the original C++ framework.

# Installation

## Prerequisites

This project requires the following dependencies:

- **[raylib](https://www.raylib.com/)** - A simple and easy-to-use library for game development
- **[raygui](https://github.com/raysan5/raygui)** - A simple and easy-to-use immediate-mode-gui library

Please install these libraries according to their respective documentation before proceeding.

## Building

A `Makefile` is included for convenience. The build process varies by platform:

### Windows

1. Ensure raylib and raygui are installed in their default locations, or update the paths in the `Makefile` accordingly.
2. Run:
   ```bash
   make
   ```
3. This will compile `controller.cpp` and produce the executable.

### Linux / Other Platforms

The provided `Makefile` is configured for Windows. For Linux or other platforms, you may need to modify the `Makefile` to match your system's raylib installation paths and compiler settings. Alternatively, you can compile directly:

```bash
gcc controller.cpp -o controller -lraylib -lraygui [additional flags as needed]
```

# Web Demo

If you want to compile the web demo you will need to first [install emscripten](https://github.com/raysan5/raylib/wiki/Working-for-Web-%28HTML5%29). Then you should be able to (on Windows) run `emsdk_env` followed by `make PLATFORM=PLATFORM_WEB`. You then need to run `wasm-server.py`, and from there will be able to access `localhost:8080/controller.html` in your web browser which should contain the demo.

# Learned Motion Matching

Most of the code and logic you can find in `controller.cpp`, with the Motion Matching search itself in `database.h`. The structure of the code is very similar to the previously mentioned [paper](https://theorangeduck.com/media/uploads/other_stuff/Learned_Motion_Matching.pdf) but not identical in all respects. For example, it does not contain some of the briefly mentioned optimizations to the animation database storage and there are no tags used to disambiguate walking and running.

## Training the Networks

The training scripts are located in the `train/` folder. See `train/README.md` for detailed documentation on the training process and model architectures.

To re-train the networks, follow these steps in order:

1. **Train the decompressor** (must be done first):
   ```bash
   cd train
   python train_decompressor.py
   ```
   This will use `database.bin` and `features.bin` from the `resources/` folder to produce `decompressor.bin`, which represents the trained decompressor network, and `latent.bin`, which represents the additional features learned for each frame in the database. It will also output some images and `.bvh` files you can use to examine the progress (as well as write Tensorboard logs to the `resources/runs` directory).

2. **Train the stepper and projector** (can be done in parallel):
   ```bash
   # Original projector (feed-forward network)
   python train_stepper.py
   python train_projector.py
   
   # OR use the diffusion-based projector (alternative implementation)
   python train_stepper.py
   python train_projector_diffusion.py
   ```
   Both scripts will output networks (`stepper.bin` and `projector.bin`) as well as some images you can use to get a rough sense of the progress and accuracy.

   **Note:** The diffusion-based projector (`train_projector_diffusion.py`) is an alternative implementation that uses a diffusion model instead of a simple feed-forward network. It maintains full compatibility with the C++ framework while potentially providing better projection quality. See `train/README_DIFFUSION.md` for more details.

All trained model files will be saved to the `resources/` directory and can be used directly with the C++ visualization program.

If you re-generate the database you will also need to re-generate the matching database `features.bin`, which is done every time you re-run the demo. Similarly if you change the weights or any other properties that affect the matching the database will need to be re-generated and the networks re-trained.

# Results

The following visualizations demonstrate the motion matching results for different methods and motion types:

## Learned Motion Matching (LMM)

### Walking
![LMM Walk](results/vis/lmm_walk.gif)

### Running
![LMM Run](results/vis/lmm_run.gif)

## Diffusion-based Learned Motion Matching (DLMM)

### Walking
![DLMM Walk](results/vis/dlmm_walk.gif)

### Running
![DLMM Run](results/vis/dlmm_run.gif)
# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2021 Daniel Holden

**Note:** The animation dataset used to generate the database is from [Ubisoft LaForge Animation Dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License, which is separate from this code's MIT License.