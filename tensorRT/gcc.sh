#!/bin/bash
# Clean the build directory to ensure a fresh build
echo "🧹 Cleaning build directory..."
rm -rf /name/forgithub/VisualFusion_libtorch/tensorRT/build
mkdir -p /name/forgithub/VisualFusion_libtorch/tensorRT/build

# Navigate to the build directory
cd /name/forgithub/VisualFusion_libtorch/tensorRT/build

# Run CMake and Make
echo "🛠️ Running CMake..."
cmake ..
echo "🏗️ Building project with Make..."
make -j$(nproc)


