#!/usr/bin/env python3

# Import statements
from model import ESRGANOptimized
import gradio as gr
import os
import time
import psutil
from pathlib import Path
from PIL import Image, ImageFilter

# Processing functions
def get_optimal_threads():
    cpu_count = psutil.cpu_count()
    if cpu_count <= 2:
        return 1
    elif cpu_count <= 4:
        return 2
    else:
        return cpu_count // 2

def resize_image(image, resize_percentage):
    if resize_percentage == 100:  # No resize needed
        return image
    
    width, height = image.size
    new_width = int(width * (resize_percentage / 100))
    new_height = int(height * (resize_percentage / 100))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def apply_gaussian_blur(image, blur_radius):
    if blur_radius == 0:  # No blur needed
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def upscale_image(input_image, scale_factor, model_path, resize_percentage, blur_radius):
    if input_image is None:
        return None, "No image provided"
    
    try:
        # Apply resize if needed
        if resize_percentage != 100 :
            input_image = resize_image(input_image, resize_percentage)
        
        # Apply blur if needed
        if blur_radius > 0 :
            input_image = apply_gaussian_blur(input_image, blur_radius)
        
        if model_path is None:
            model_path = "4x-ClearRealityV1-fp32-opset17.onnx"
        # Initialize model
        if scale_factor is None:
            scale_factor = 1
        tile_size = 256 * scale_factor
        if "fp16" in model_path:
            model = ESRGANOptimized(
                model_path=model_path,
                tile_size=tile_size,
                model_input_size=256,
                scale=4,
                num_threads=get_optimal_threads(),
                overlap_size=8,
                use_fp16=True
            )
        else:
            model = ESRGANOptimized(
                model_path=model_path,
                tile_size=tile_size,
                model_input_size=256,
                scale=4,
                num_threads=get_optimal_threads(),
                overlap_size=8,
                use_fp16=False
            )
        
        # Generate a unique filename for the input image
        input_path = f"/tmp/input_{time.time():.0f}.png"
        input_image.save(input_path)
        
        # Create output filename with '_upscaled' suffix
        base_name = Path(input_path).stem
        output_filename = f"{base_name}_upscaled.png"
        output_path = f"/tmp/{output_filename}"
        
        start_time = time.time()
        
        if model.process_image(input_path, output_path):
            process_time = time.time() - start_time
            os.remove(input_path)  # Cleanup
            return output_path, f"Success! Processing time: {process_time:.2f}s"
        else:
            return None, "Processing failed"
            
    except Exception as e:
        return None, f"Error occurred: {str(e)}"

# Create and launch interface
if __name__ == "__main__":
    # Upscale factors
    scales = [1, 2, 3, 4, 6, 8]
    scale_dropdown = gr.Dropdown(scales, label="Upscale Factor")
    
    # Model selection
    model_paths = {
        "4xNomos2_realplksr_dysample_256_int8_fullyoptimized.onnx": "Fully Optimized Model",
        "4x-ClearRealityV1-fp32-opset17.onnx": "ClearReality Model",
        "4x-UltraSharp-fp16-opset17.onnx": "UltraSharp Model"
    }
    model_dropdown = gr.Dropdown(list(model_paths.values()), label="Model")
    
    # Resize options
    resize_options = [25, 50, 75, 100]  # 100 means no resize
    resize_dropdown = gr.Dropdown(
        choices=resize_options,
        value=100,
        label="Resize Percentage"
    )
    
    # Blur options
    blur_slider = gr.Slider(
        minimum=0,
        maximum=1,
        step=0.1,
        value=0,
        label="Gaussian Blur Radius"
    )
    
    iface = gr.Interface(
        fn=lambda input_image, scale_factor, model_name, resize_percent, blur_rad: upscale_image(
            input_image,
            scale_factor,
            [k for k, v in model_paths.items() if v == model_name][0],
            resize_percent,
            blur_rad
        ),
        inputs=[
            gr.Image(type="pil"),
            scale_dropdown,
            model_dropdown,
            resize_dropdown,
            blur_slider
        ],
        outputs=[
            gr.Image(type="filepath", label="Upscaled Image"),
            gr.Textbox(label="Status")
        ],
        title="Image Upscaler",
        description="Upload an image, select the upscale factor, choose the model, and apply optional resize and blur effects"
    )
    
    iface.launch(
        share=True,
        server_name="0.0.0.0",
        debug=True,
        server_port=7860
    )