# Image Processing Transformations Reference

This document serves as a reference for common image processing filters and the underlying algorithms/math that power them. Because the application utilizes a `wgpu` stack, these formulas translate into mathematically parallelized WebGPU (WGSL) operations applied per pixel globally or isolated by an alpha-mask.

### 1. Basic Color Adjustments
* **Brightness / Exposure**: A simple arithmetic operation. Brightness is usually a constant offset added to all RGB channels equally (e.g., `R + offset`). Exposure acts slightly differently as a multiplier (e.g., `R * scale`), stretching the existing values.
* **Contrast**: Scales pixel values outward from the middle gray value. Mathematically: `(Value - 0.5) * ContrastFactor + 0.5`. This pushes darks darker and lights brighter.
* **Saturation**: The shader calculates the perceptual "gray" luminance of the pixel, then linearly interpolates between that pure gray value and the original RGB color. Moving away from gray increases saturation.
* **Vibrance**: Applies a non-linear scaling factor, boosting the saturation of muted colors *more* than colors that are already highly saturated, often mathematically isolating skin tones.

### 2. Conversions & Toning
* **Black & White (Grayscale / Luminance)**: Standardized weighted addition based on human eye perception (ITU-R BT.709 standard). `Luminance = (0.2126 * R) + (0.7152 * G) + (0.0722 * B)`.
* **Sepia**: Achieved via a specific matrix multiplication against the RGB channels. It re-weights the colors to artificially boost warm, brownish-yellow tones imitating photographic degradation.
* **Invert (Negative)**: `1.0 - Channel`. White becomes black, reds become cyan, etc.

### 3. Spatial Filters (Using Gaussian Kernels/Matrices)
_Note: These require evaluating neighboring pixels, not just the isolated pixel being processed._
* **Gaussian Blur**: A "convolution matrix". It looks at surrounding pixels and averages them together using a bell-curve (Gaussian) weighting system. For shader performance, it is generally done in two passes (a separated horizontal pass, then a vertical pass).
* **Unsharp Mask (Sharpening)**: Creates a blurred copy of the image, subtracts the blurred copy from the original image to isolate just the high-contrast edges, and then adds those edges mathematically back into the original image context.
* **Edge Detection (e.g., Sobel Operator)**: Applies a matrix kernel that calculates the gradient between neighboring pixels. High rates of change represent an edge, mapping cleanly to white.

### 4. Advanced Tonal Mapping
* **Shadows / Highlights Recovery**: Applies non-linear curves/gamma correction strictly limited to specific luminosity thresholds. If a pixel's luminance is below 0.3 (dark), it gets boosted smoothly.
* **Temperature & Tint (White Balance)**: Adjusts colors along two standard axes. Temperature modifies the Blue-to-Yellow axis, and Tint modifies the Green-to-Magenta axis.

### 5. Stylization
* **Cartoon**: A toon-filter pipeline implemented as five sequential compute passes. See
  `specs/multi-pass-plan.md` § "Cartoon" for the full design rationale.
  1. **Smooth (H + V)**: A separable Gaussian blur (σ = 1.5% of the longer image dimension)
     softens the image, removing fine texture while preserving broad color regions.
  2. **Quantize**: Posterizes the smoothed image into N discrete levels per channel in
     linear-light space. Banding boundaries fall at energy-uniform intervals (differs visibly
     from sRGB-gamma quantization such as Photoshop Posterize).
  3. **Edges**: A 3×3 Sobel operator computes gradient magnitude on the Rec.709 luma of the
     *original* source image (not the smoothed one, which has had its edges erased). The
     resulting mask is shaped by a user-controlled threshold and softness ramp:
     `edge = smoothstep(threshold, threshold + softness, sobel_magnitude)`.
  4. **Combine**: Blends the original and posterized images by `Strength`, then darkens by
     the edge mask: `out = mix(src, quant, strength) * (1 - edge_darkness * edge_mask)`.
  Sliders: Strength, Levels, Edge Threshold, Edge Softness, Edge Darkness.
