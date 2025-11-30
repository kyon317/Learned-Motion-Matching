# Motion Matching & Code vs Data Driven Displacement

This repository contains my implementation of Learned Motion Matching, based on the original work described in [this article](https://theorangeduck.com/page/code-vs-data-driven-displacement) and [this paper](https://theorangeduck.com/page/learned-motion-matching). I also implemented a diffusion based network to replace the original Projector network, you can check details in the Training section.

This implementation includes custom training scripts in the `train/` folder, including an alternative diffusion-based projector network, while maintaining full compatibility with the original C++ framework.

# Installation

This demo uses [raylib](https://www.raylib.com/) and [raygui](https://github.com/raysan5/raygui) so you will need to first install those. Once installed, the demo itself is a pretty straight forward to make - just compile `controller.cpp`.

I've included a basic `Makefile` which you can use if you are using raylib on Windows. You may need to edit the paths in the `Makefile` but assuming default installation locations you can just run `Make`.

If you are on Linux or another platform you will probably have to hack this `Makefile` a bit.

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

The data required if you want to regenerate the animation database is from [this dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) which is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (unlike the code, which is licensed under MIT).

If you re-generate the database you will also need to re-generate the matching database `features.bin`, which is done every time you re-run the demo. Similarly if you change the weights or any other properties that affect the matching the database will need to be re-generated and the networks re-trained.
