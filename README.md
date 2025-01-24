# Multithreaded SuperPoint Inference for AMD Xilinx Kria KR260

This repository provides a multithreaded implementation of the SuperPoint feature detection and description algorithm, tailored to run on AMD Xilinx Kria KR260 devices using the Vitis AI DPU. It addresses the challenge of unsupported operators on the DPU by offloading those layers to the CPU in a pipeline that is optimized with multithreading.

---

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Repository Structure](#repository-structure)  
4. [Build Instructions](#build-instructions)  
5. [Usage](#usage)  
6. [Profiling and Performance Notes](#profiling-and-performance-notes)  
7. [Future Improvements](#future-improvements)  
8. [License](#license)

---

## Overview
SuperPoint is a deep learning-based method for detecting keypoints and computing descriptors in an image. It features:
- A **heatmap head** for identifying interest points.
- A **descriptor head** for generating vector descriptors around each interest point.

On the Kria KR260, many standard convolutional layers can be accelerated via the Vitis AI DPU. However, certain operators or layers (e.g., specialized post-processing or interpolation layers) may not be directly supported by the DPU. In this code, these layers are offloaded to the CPU and run in **parallel** (using multi-threading) alongside the DPU execution to reduce bottlenecks.

---

## Features
- **DPU Acceleration**  
  Leverages Vitis AI for hardware-accelerated inference of SuperPoint’s convolutional layers.
- **CPU Offloading for Unsupported Layers**  
  Special operators, such as custom interpolation or NMS (non-maximum suppression), run on the CPU.
- **Multithreaded Pipeline**  
  CPU-bound tasks like descriptor extraction or non-maximum suppression are parallelized to improve overall throughput.
- **Modular Codebase**  
  Easy to extend or replace components (e.g., different feature extraction steps, alternative post-processing methods).

---

## Repository Structure

```
.
├── CMakeLists.txt
├── include/
│   └── (Header files for SuperPoint)
├── src/
│   ├── superpoint.cpp          # Core SuperPoint implementation
│   ├── demo_superpoint.cpp     # Demo application (entry point)
│   └── (Other utility/source files)
├── models/
│   └── (Place your .xmodel or model files here)
└── README.md                   # This file
```

- **`superpoint.cpp`**: Implements the SuperPoint class, handling DPU input/output and post-processing steps.  
- **`demo_superpoint.cpp`**: Example usage that loads an image, runs the inference pipeline, and outputs the processed result.  
- **`CMakeLists.txt`**: Build configuration.  

---

## Build Instructions

1. **Install Dependencies**
   - **Vitis AI** runtime and libraries for the Kria KR260.  
   - **OpenCV** (version 4.5.4 or later).  
   - **Glog**, **Threads**, and other libraries required by Vitis AI.  
   - Ensure `CMake` (version 3.10 or later) is installed.

2. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/Multithreaded-SuperPoint-Kria.git
   cd Multithreaded-SuperPoint-Kria
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
1. **Run the Demo**
   ```bash
   ./demo <model_name> <test_image>
   ```
   - `<model_name>`: Path to your compiled `.xmodel` (e.g., `superpoint_tf.xmodel`).
   - `<test_image>`: Path to a grayscale image (e.g., `test.jpg`).

2. **Expected Output**
   - Keypoints are drawn on the image, and the processed image is saved as `result_superpoint_0.jpg` (or similar) in the working directory.
   - The console shows timing results for the total inference (including CPU post-processing).

---

## Profiling and Performance Notes

### Profiling Setup
- The code can be compiled with profiling flags (e.g., `-pg`) and analyzed with `gprof` or other profiling tools.

### Key Bottlenecks
According to our profiling report (see [Performance Analysis and Optimization Report](#) in the `superpoint.cpp` doc comments):
- **Descriptor Extraction (`grid_sample`)** on the CPU can consume over 30% of runtime for large batches.
- **CPU-based Post-Processing** (`verifyOutput` function) is the next major time consumer.
- **DPU Inference** is not captured in CPU-side profiling as it runs on dedicated hardware.

### Multithreading Approach
- **Thread Pool**: The CPU-bound tasks (e.g., `grid_sample`, NMS) are distributed across multiple threads.
- **Pipeline Overlap**: Preprocessing, DPU execution, and CPU post-processing can be overlapped for successive batches to reduce idle time.

### Recommended Optimizations
- **SIMD / NEON Intrinsics**: Speed up interpolation and normalization on the CPU.
- **Batch Pipelining**: Feed the next batch to the DPU while post-processing the results of the previous batch.
- **Data Locality**: Minimize data copies and memory allocations; allocate buffers once and reuse them.

---

## Future Improvements
1. **Dynamic Load Balancing**: Automatically distribute CPU tasks based on real-time load to avoid idle cores.
2. **Enhanced Post-Processing**: Explore approximate Non-Maximum Suppression or further vectorization for faster keypoint filtering.
3. **Reduced Memory Footprint**: Streamline image buffers and descriptors to support more images concurrently without running out of memory.
4. **Integration of Additional Modules**: Add image enhancement or other pre/post-processing techniques for better feature robustness.

---

### Thank You!
We hope this repository helps you accelerate SuperPoint inference on your Kria KR260 device. Feel free to open an issue or contribute improvements!  