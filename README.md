# VisualFusion

🔥 **Real-Time EO-IR Image Alignment and Fusion System**

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![LibTorch](https://img.shields.io/badge/LibTorch-2.0+-orange.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.15+-purple.svg)](https://onnx.ai/)

## 🚀 Overview

VisualFusion LibTorch is a computer vision system for **EO-IR (Electro-Optical/Infrared) image alignment and fusion**. It uses deep learning models to detect feature points and compute homography matrices for accurate image registration, then creates fused outputs with advanced edge-preserving algorithms.

### 🎨 Demo

Input images and fusion result:

| EO (Electro-Optical) | IR (Infrared) |
|:---:|:---:|
| ![EO Image](demo/demo_EO.jpg) | ![IR Image](demo/demo_IR.jpg) |

<div align="center">
  <img src="demo/output.jpg" alt="Fusion Result" width="100%">
  <br>
  <em>Fusion Result</em>
</div>

### ✨ Key Features

- ⚙️ **Video & Image Cropping**: Support for VideoCut and PictureCut parameters
- 🎯 **Deep Learning Feature Detection**: Uses SemLA model for keypoint detection and matching
- 🖼️ **EO-IR Image Fusion**: Seamless fusion with shadow enhancement and edge preservation  
- 📐 **RANSAC Homography**: Robust estimation with outlier filtering
- 🎛️ **Homography Smoothing**: Temporal consistency with configurable smoothing parameters
- 📊 **Performance Timing**: Built-in timer analysis for each processing stage

## 🏗️ Architecture

```
VisualFusion_libtorch/
├── IR_Convert_v21_libtorch/    # LibTorch implementation 
├── Onnx/                       # ONNX Runtime implementation
├── tensorRT/                   # TensorRT implementation
└── convert_to_libtorch/        # Model conversion utilities
```

### 🔧 Supported Inference Engines

| Engine | Model Format | Device | Key Features |
|--------|--------------|--------|-------------|
| **LibTorch** | `.zip` (TorchScript) | CPU/CUDA | Dynamic shapes, flexible deployment |
| **ONNX Runtime** | `.onnx` | CPU/CUDA | Optimized CUDA inference, cross-platform |
| **TensorRT** | `.engine` | CUDA | Optimized CUDA inference, FP16 support |

## 📋 Requirements

- **OS**: Ubuntu 20.04+
- **C++ Compiler**: GCC 9+, CMake 3.18+
- **OpenCV**: 4.5+ (core, imgcodecs, highgui, calib3d, videoio)
- **LibTorch**: 11.7 (LibTorch version only)
- **ONNX Runtime**: 1.16.8 (ONNX version only)
- **CUDA & TensorRT**: 11.4+ & 8.6.1.6+ (TensorRT version only)

## 🛠️ Installation & Usage

### LibTorch Version

```bash
cd IR_Convert_v21_libtorch

# Build and run
bash gcc.sh && ./build/out
```

### ONNX Runtime Version  

```bash
cd Onnx

# Build and run
bash gcc.sh && ./build/out

```

### TensorRT Version

```bash
cd tensorRT

# Build and run
bash gcc.sh && ./build/out
```

#### TensorRT Model Conversion

```bash
# Convert ONNX model to TensorRT engine
cd convert_to_libtorch
python export_onnx2tensorRT.py --fp16 --opset 12
```

## ⚙️ Configuration

### TensorRT Configuration

```json
{
    "input_dir": "/path/to/input",
    "output_dir": "/path/to/output", 
    "output": true,
    
    "device": "cpu",
    "pred_mode": "fp32",
    "model_path": "/path/to/model.trt",
    
    "output_width": 320,
    "output_height": 240,
    "pred_width": 320, 
    "pred_height": 240,
    
    "VideoCut": true,
    "Vcut_x": 870,
    "Vcut_y": 235, 
    "Vcut_w": 2020,
    "Vcut_h": 1680,
    
    "PictureCut": true,
    "Pcut_x": 220,
    "Pcut_y": 0,
    "Pcut_w": 1920,
    "Pcut_h": 1080,
    
    "fusion_shadow": true,
    "fusion_edge_border": 2,
    "fusion_threshold_equalization": 128,
    "fusion_threshold_equalization_low": 72,
    "fusion_threshold_equalization_high": 192,
    "fusion_threshold_equalization_zero": 64,
    "fusion_interpolation": "cubic",
    
    "perspective_check": true,
    "perspective_distance": 10,
    "perspective_accuracy": 0.85,
    
    "align_distance_last": 15.0,
    "align_distance_line": 10.0,
    "align_angle_mean": 10.0,
    "align_angle_sort": 0.7,
    
    "smooth_max_translation_diff": 80.0,
    "smooth_max_rotation_diff": 0.05,
    "smooth_alpha": 0.05
}
```

### Core Parameters

```json
{
    "input_dir": "/path/to/input",
    "output_dir": "/path/to/output", 
    "output": true,
    
    "device": "cuda",
    "pred_mode": "fp32",
    "model_path": "./model/SemLA_jit_cuda.zip",
    
    "output_width": 320,
    "output_height": 240,
    "pred_width": 320, 
    "pred_height": 240
}
```

### Video/Image Cropping

```json
{
    "VideoCut": true,
    "Vcut_x": 870,
    "Vcut_y": 235, 
    "Vcut_w": 2020,
    "Vcut_h": 1680,
    
    "PictureCut": true,
    "Pcut_x": 220,
    "Pcut_y": 0,
    "Pcut_w": 1920,
    "Pcut_h": 1080
}
```

### Fusion Settings

```json
{
    "fusion_shadow": true,
    "fusion_edge_border": 2,
    "fusion_threshold_equalization": 128,
    "fusion_threshold_equalization_low": 72,
    "fusion_threshold_equalization_high": 192,
    "fusion_threshold_equalization_zero": 64,
    "fusion_interpolation": "cubic"
}
```

### Perspective & Alignment

```json
{
    "perspective_check": true,
    "perspective_distance": 10,
    "perspective_accuracy": 0.85,
    
    "align_distance_last": 15.0,
    "align_distance_line": 10.0,
    "align_angle_mean": 10.0,
    "align_angle_sort": 0.7
}
```

### Homography Smoothing

```json
{
    "smooth_max_translation_diff": 80.0,
    "smooth_max_rotation_diff": 0.05,
    "smooth_alpha": 0.05
}
```

## 📁 Input Format

The system expects paired EO-IR images with `_EO` and `_IR` suffixes:

```
input/
├── scene_001_EO.jpg
├── scene_001_IR.jpg
├── scene_002_EO.jpg  
└── scene_002_IR.jpg
```

## 🎮 Processing Pipeline

1. **Input Processing**: EO-IR image pairs (`_EO`/`_IR` naming) or video files
2. **Preprocessing**: VideoCut/PictureCut → Resize → Grayscale conversion
3. **Feature Detection**: SemLA model inference (1200 fixed keypoints, use first `leng1` valid points)
4. **Homography**: RANSAC estimation with temporal smoothing
5. **Enhancement**: Canny edge detection with perspective transformation
6. **Fusion**: Multi-threshold shadow enhancement with configurable interpolation
7. **Output**: Feature visualization + fused result with timing analysis

## 🔍 Algorithm Components

### Deep Learning Feature Matching

**SemLA Model Architecture**
- **Input**: Grayscale image pairs (320×240)
- **Output**: Fixed 1200 keypoint pairs with validity count
- **Feature Spacing**: Minimum 8px distance between keypoints
- **Padding Strategy**: Insufficient points padded to (0,0) coordinates
- **Valid Count**: `leng1` parameter indicates actual detected keypoints
- **Usage**: Extract first `leng1` points as true feature matches

**RANSAC Homography Estimation**  
- **Algorithm**: Random Sample Consensus with 8.0px threshold
- **Minimum Points**: 4 correspondences required
- **Confidence**: 0.99 success probability with 1000 max iterations
- **Quality Validation**: Determinant bounds and inlier ratio checks

**Temporal Homography Smoothing**
- **Translation**: Maximum 80.0px difference threshold
- **Rotation**: Maximum 0.05 radian difference threshold  
- **Smoothing**: Exponential moving average (α=0.05)
- **Fallback**: Automatic reset after 3 consecutive rejections

**Image Fusion & Enhancement**
- **Edge Detection**: Multi-threshold Canny with 2px border
- **Shadow Enhancement**: Multi-level histogram equalization (thresholds: 128/72/192/64)
- **Interpolation**: Linear or cubic resampling options
- **Blending**: Edge-aware alpha composition

### Image Fusion
- **Edge Enhancement**: Canny edge detection with adaptive thresholds
- **Shadow Processing**: Histogram equalization with multiple threshold levels
- **Interpolation**: Linear or cubic resampling options
- **Blending**: Alpha composition with edge-aware weights

## 📊 Performance

### Typical Performance (320×240 resolution)
- **LibTorch**: 124ms (CPU), 13ms (CUDA) 
- **ONNX Runtime**: 215ms (CPU), 11ms (CUDA)
- **TensorRT**: 5.6ms (CPU), 3.2ms (CUDA)
### Timing Components
All versions provide detailed timing for: resize, grayscale conversion, model inference, homography computation, edge detection, perspective transform, and fusion operations.

## 🐛 Troubleshooting

### Common Issues
```bash
# Check model files and permissions
ls -la model/
# Monitor memory usage (CPU/GPU)
htop && nvidia-smi
# Switch to CPU mode if needed
sed -i 's/"device": "cuda"/"device": "cpu"/' config/config.json
```

### Version-Specific
```bash
# LibTorch: Check installation and CUDA compatibility
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# ONNX Runtime: Verify model and runtime
python -c "import onnxruntime; onnxruntime.InferenceSession('model.onnx')"

# TensorRT: Regenerate engine and test loading
python export_onnx2tensorRT.py --onnx model.onnx --trt model.trt
```

### Parameter Tuning
- **Accuracy**: Increase `perspective_accuracy` (0.85→0.95), use cubic interpolation
- **Speed**: Use TensorRT with FP16, reduce resolution, enable frame skipping  
- **Stability**: Decrease `smooth_alpha` (0.05→0.02), increase smoothing thresholds

## 🔧 Development

### Build System
Each version uses CMake with a `gcc.sh` build script:
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release  
make -j$(nproc)
```

### Code Architecture
```
lib_image_fusion/
├── include/core_image_align_*.h    # Version-specific inference interfaces
├── src/core_image_align_*.cpp      # LibTorch/ONNX/TensorRT implementations
├── include/core_image_fusion.h     # Image blending and enhancement
└── src/*.cpp                       # Shared processing modules
```

### Key Features
- **Unified Interface**: All versions implement the same `align()` function
- **Modular Design**: Swap inference engines without changing main pipeline
- **Thread Safety**: Multi-threaded processing support
- **Error Handling**: Comprehensive logging and graceful fallbacks
- **Hot Reload**: Runtime configuration updates via JSON
