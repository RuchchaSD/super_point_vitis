#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Clean up previous CMake configuration files
rm -rf CMakeCache.txt CMakeFiles/ cmake_install.cmake

# Configure with CMake in Release mode
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF

# Build the project
make -j$(nproc)

echo "Build completed successfully and libraries are in lib/"

# Return to original directory
cd ..
