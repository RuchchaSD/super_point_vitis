# Multithreaded SuperPoint Inference for AMD Xilinx Kria KR260

This repository provides a multithreaded implementation of the SuperPoint feature detection and description algorithm, tailored to run on AMD Xilinx Kria KR260 devices using the Vitis AI DPU. It addresses the challenge of unsupported operators on the DPU by offloading those layers to the CPU in a pipeline that is optimized with multithreading.

---

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Repository Structure](#repository-structure)  
4. [Build Instructions](#build-instructions)  
5. [Usage](#usage)
   - [Running the Demo (`demo`)](#running-the-demo-demo)
   - [Running Continuous Processing (`demo_continuous`)](#running-continuous-processing-demo_continuous)
   - [Extracting Features (`extract_features`)](#extracting-features-extract_features)
6. [Profiling and Performance Notes](#profiling-and-performance-notes)  
7. [Future Improvements](#future-improvements)  
8. [License](#license)

---

## Overview
SuperPoint is a deep learning-based method for detecting keypoints and computing descriptors in an image. This implementation is optimized for the AMD Xilinx Kria KR260, utilizing its Vitis AI DPU for accelerating convolutional layers.

The core architecture (`SuperPointFast.cpp`) employs a pipelined, multithreaded approach:
1.  **Pre-processing**: Input images are resized, converted to grayscale, and prepared for the DPU. This stage can run on multiple CPU threads.
2.  **DPU Inference**: The SuperPoint model's convolutional layers are executed on the DPU. This implementation uses a fixed pool of DPU runners (typically 4, as defined by `NUM_DPU_RUNNERS` in `SuperPointFast.h`) to handle concurrent inference tasks.
3.  **Post-processing**: The DPU output (heatmap and descriptors) is processed on the CPU. This involves:
    *   Softmax on the heatmap (can be hardware-accelerated if `HW_SOFTMAX` is defined).
    *   Keypoint detection (NMS - Non-Maximum Suppression).
    *   Descriptor L2 normalization and extraction using bilinear sampling.
    This stage also leverages multiple CPU threads.

Unsupported DPU operations and the pre/post-processing stages are efficiently managed on the CPU using separate thread pools to work in parallel with DPU execution, minimizing bottlenecks and maximizing throughput. Thread-safe queues are used to pass data between these pipeline stages.

---

## Features
- **DPU Acceleration**: Leverages Vitis AI for hardware-accelerated inference of SuperPoint's convolutional layers using a pool of DPU runners.
- **CPU Offloading for Unsupported Layers & Processing**: Pre-processing (resizing, grayscale conversion), and post-processing tasks (softmax, NMS, descriptor extraction, L2 normalization) run on the CPU.
- **Multithreaded Pipeline**:
    - Dedicated thread pools for pre-processing and post-processing stages.
    - Fixed number of DPU runners for concurrent DPU task execution.
    - Thread-safe queues manage data flow between pre-processing, DPU inference, and post-processing stages, enabling pipelined execution.
- **Modular Codebase**:
    - Core logic in `SuperPointFast.cpp` and `SuperPointFast.h`.
    - Utility functions and types in `include/utils/`.
- **Flexible Execution Modes**: Supports single image processing, batch processing, and continuous processing from an input queue.

---

## Repository Structure

```
.
├── CMakeLists.txt
├── include/
│   ├── SuperPointFast.h
│   └── utils/              # Helper utilities (ThreadSafeQueue, SuperPointTypes, etc.)
├── src/
│   ├── SuperPointFast.cpp  # Core SuperPoint multithreaded implementation
│   ├── demo_superpoint.cpp # Demo application for single/batch image processing
│   ├── demo_continuous.cpp # Demo application for continuous (stream) processing
│   ├── extract_features.cpp# Application to extract and save features
│   └── (Other utility/source files e.g. FeatureIO.cpp)
├── models/                   # Directory for .xmodel files (e.g., superpoint_tf.xmodel)
├── results/                  # Default output directory for demo applications
│   ├── benchmark/
│   └── continuous/
├── feature_outputs/          # Default output directory for feature extraction
├── run_superpoint.sh         # Script to run the SuperPoint demo
├── run_continuous.sh       # Script to run the continuous processing demo
├── extract_features.sh     # Script to run feature extraction
└── README.md                 # This file
```

- **`SuperPointFast.cpp` / `SuperPointFast.h`**: Implements the `SuperPointFast` class, handling DPU input/output, multithreaded pre/post-processing, and continuous processing logic.
- **`demo_superpoint.cpp`**: Example usage that loads an image (or multiple for batch), runs the inference pipeline, and outputs the processed result.
- **`demo_continuous.cpp`**: Example usage for processing a continuous stream of images from a directory.
- **`extract_features.cpp`**: Application to load images, run SuperPoint inference, and save keypoints and descriptors to specified output directories.
- **`CMakeLists.txt`**: Build configuration.
- **Shell Scripts (`run_*.sh`)**: Convenience scripts to execute the different demo applications with pre-configured or environment-variable-driven parameters.

---

## Build Instructions

1. **Prerequisites & Dependencies**
   Ensure the following are installed and configured on your Kria KR260 or build environment:
   - **Vitis AI SDK**: Including runtime, libraries, and DPU drivers. Specifically, the build process requires:
     - `VART` (Vitis AI Runtime)
     - `UNILOG` (Vitis AI Logging)
     - `XIR` (Xilinx Intermediate Representation)
     - `vitis_ai_library` (specifically `dpu_task` and `math` components)
   - **OpenCV**: (e.g., version 4.5.4 or later, as found by `find_package(OpenCV REQUIRED)`).
   - **Glog**: Google Logging Library (e.g., version 0.5.0 or later, as found by `find_package(Glog REQUIRED)`).
   - **CMake**: (Version 3.10 or later).
   - **Standard C++ Compiler** supporting C++17.
   - **Threads**: POSIX Threads library (`find_package(Threads REQUIRED)`).

   The `CMakeLists.txt` file will attempt to locate these dependencies. You may need to set `CMAKE_PREFIX_PATH` or other environment variables if they are installed in non-standard locations. For example, `GLOG_PATH` is explicitly set in the `CMakeLists.txt` for Glog.

2. **Clone the Repository**
   ```bash
   git [clone https://github.com/YourUsername/Multithreaded-SuperPoint-Kria.git](https://github.com/RuchchaSD/super_point_vitis.git)
   cd super_point_vitis
   ```

3. **Configure and Build**
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

4. **Deploy to the Target**  
   Copy the `demo` executable, the compiled libraries, and your `.xmodel` files to the Kria KR260 target (if not building natively on the device).

---

## Usage

This project provides several executables and helper scripts to run the SuperPoint inference.

### General Notes:
-   **Models**: Place your compiled `.xmodel` files (e.g., `superpoint_tf.xmodel`, `compiled_SP_by_H.xmodel`) in a known location. The scripts often default to a path within the project or expect it as an argument.
-   **Output**: Results (images with keypoints, feature files) are typically saved to the `results/` or `feature_outputs/` directories, or a custom path if specified.
-   **Environment**: Ensure the PYNQ environment is sourced if running on the Kria board: `source /etc/profile.d/pynq_venv.sh` (this is included in the provided `.sh` scripts).

### 1. Running the Demo (`demo` via `run_superpoint.sh`)

The `demo` executable processes a single image or runs multiple iterations on the same image for benchmarking. The `run_superpoint.sh` script simplifies its execution.

**Script Usage:**
```bash
./run_superpoint.sh
```

**Key Environment Variables for `run_superpoint.sh` (can be set before running):**
-   `IMPLEMENTATION`: Set to `multi` (default) for the multithreaded `SuperPointFast` or `single` (not typically used with `SuperPointFast`).
-   `RUNNERS`: Number of DPU runners/threads (default: `10`, though `SuperPointFast` uses a fixed internal pool for DPU runners, this might influence CPU threads for pre/post-processing in some contexts or be a legacy from older versions. The `SuperPointFast` constructor takes `num_threads` which dictates pre/post processing threads).
-   `ITERATIONS`: Number of inference iterations (default: `100`).
-   `MODEL`: Path to the `.xmodel` file (default: `compiled_SP_by_H.xmodel`).
-   `IMAGE`: Path to the input image (default: a test image in `temp/imgs/`).
-   `OUTPUT_DIR`: Directory to store result images (default: `./results`).

**Direct Executable Usage (example):**
```bash
./build/demo -t <num_threads> <model_name> <test_image>
# Example for SuperPointFast (num_threads for CPU pre/post processing)
./build/demo -t 4 compiled_SP_by_H.xmodel test.jpg 
```
- Output images like `result_superpoint_0.jpg` are saved in the current working directory or `OUTPUT_DIR` if using the script.

### 2. Running Continuous Processing (`demo_continuous` via `run_continuous.sh`)

The `demo_continuous` executable processes all images from a specified input directory using the `SuperPointFast` continuous run mode.

**Script Usage:**
```bash
./run_continuous.sh
```

**Key Environment Variables for `run_continuous.sh`:**
-   `THREADS`: Number of pre/post-processing threads (default: `4`). Passed as `-t` to the executable.
-   `MODEL`: Path to the `.xmodel` file (default: `compiled_SP_by_H.xmodel`).
-   `INPUT_DIR`: Directory containing input images (default: `temp/imgs/`).
-   `OUTPUT_DIR`: Directory to store result images (default: `./results/continuous`).
-   `FILE_EXT`: File extension of images to process (default: `png`). Passed as `-f` to the executable.

**Direct Executable Usage (example):**
```bash
./build/demo_continuous -t <num_threads> -f <file_ext> <model_name> <input_dir> <output_dir>
# Example:
./build/demo_continuous -t 4 -f png compiled_SP_by_H.xmodel temp/imgs/ ./results/continuous/
```

### 3. Extracting Features (`extract_features` via `extract_features.sh`)

The `extract_features` executable processes images from an input directory and saves the detected keypoints and descriptors to separate files in a structured output directory.

**Script Usage:**
```bash
./extract_features.sh
```

**Key Environment Variables for `extract_features.sh`:**
-   `THREADS`: Number of pre/post-processing threads (default: `4`). Passed as `-t` to the executable.
-   `MODEL`: Path to the `.xmodel` file (default: `superpoint_tf.xmodel`).
-   `INPUT_DIR`: Directory containing input images (default: `temp/datasets/MH_01_easy/mav0/cam0/data`).
-   `OUTPUT_DIR`: Base directory to store feature outputs (e.g., `./feature_outputs/SP_TF`). Keypoints and descriptors will be saved in subfolders like `kpts/` and `desc/`.
-   `FILE_EXT`: File extension of images to process (default: `png`). Passed as `-f` to the executable.

**Direct Executable Usage (example):**
```bash
./build/extract_features -t <num_threads> -f <file_ext> <model_name> <input_dir> <output_dir>
# Example:
./build/extract_features -t 4 -f png superpoint_tf.xmodel temp/imgs/ ./feature_outputs/SP_TF_direct/
```
- This will create `kpts/` and `desc/` subdirectories within the specified output path.

---

## Profiling and Performance Notes

### Profiling Setup
- The code can be compiled with profiling flags (e.g., by uncommenting `-pg` flags in `CMakeLists.txt`) and analyzed with `gprof` or other profiling tools.
- Vitis AI provides its own profiling tools (`vai_profiler`) that can be used to analyze DPU performance and the overall application timeline. The `vitis::ai::profiling` macros (`__TIC__`, `__TOC__`) are used throughout `SuperPointFast.cpp` to mark sections for detailed timing.

### Key Considerations in `SuperPointFast`
- **Fixed DPU Runners**: `SuperPointFast` uses a constant `NUM_DPU_RUNNERS` (typically 4) for DPU inference. This is designed to saturate the DPU capacity.
- **CPU-bound Tasks**: Pre-processing and post-processing (softmax, NMS, descriptor extraction) are parallelized across a configurable number of CPU threads (`num_threads_` passed to the `SuperPointFast` constructor).
  - **Softmax**: Can be DPU-accelerated (`vitis::ai::softmax`) if `HW_SOFTMAX` is defined, otherwise runs on CPU.
  - **NMS (Non-Maximum Suppression)**: A critical CPU-bound step.
  - **Descriptor Extraction**: Involves `L2_normalization` and `bilinear_sample`, which are CPU-intensive.
- **DPU Inference**: While the DPU itself is fast, data transfers to/from the DPU and synchronization can add overhead.

### Multithreading Approach in `SuperPointFast`
- **Pipeline Stages**: The system is divided into three main concurrent stages: pre-processing, DPU inference, and post-processing.
- **Thread Pools**:
    - Pre-processing uses a pool of `num_threads_` threads.
    - DPU inference stage manages its own fixed set of DPU runners (internally, these might map to threads or dedicated hardware contexts).
    - Post-processing uses a pool of `num_threads_` threads.
- **Thread-Safe Queues**: `ThreadSafeQueue` instances are used to pass data (image tasks, DPU results) between these stages, allowing them to operate largely independently and in a pipelined fashion.
    - `task_queue`: Holds pre-processed data ready for DPU.
    - `result_queue`: Holds DPU output ready for post-processing.
- **Continuous Mode**: The `run(ThreadSafeQueue<InputQueueItem>&, ThreadSafeQueue<SuperPointResult>&)` overload in `SuperPointFast` is designed for continuous image stream processing, where the entire pipeline runs in a dedicated background thread (`pipeline_thread_`).

### Potential Optimizations (Some already present or considered)
- **SIMD / NEON Intrinsics**: The code structure includes `#ifdef ENABLE_NEON` guards, suggesting NEON optimizations can be added/enabled for CPU-bound tasks like image conversion or parts of post-processing.
- **Batch Pipelining**: The inherent design with queues allows for batch pipelining: the DPU can process one batch while the CPU pre-processes the next and post-processes the previous.
- **Data Locality & Copy Reduction**: `std::memcpy` is used for efficient data transfers to/from DPU buffers. Efforts are made to resize vectors and allocate memory appropriately to minimize reallocations.

---

## Future Improvements
<!-- 1. **Dynamic Load Balancing**: Automatically distribute CPU tasks based on real-time load to avoid idle cores.
2. **Enhanced Post-Processing**: Explore approximate Non-Maximum Suppression or further vectorization for faster keypoint filtering.
3. **Reduced Memory Footprint**: Streamline image buffers and descriptors to support more images concurrently without running out of memory.
4. **Integration of Additional Modules**: Add image enhancement or other pre/post-processing techniques for better feature robustness. -->

---

### Thank You!
We hope this repository helps you accelerate SuperPoint inference on your Kria KR260 device. Feel free to open an issue or contribute improvements!  
