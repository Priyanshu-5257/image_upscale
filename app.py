#!/usr/bin/env python3

# Import statements
from model import ESRGANOptimized
import gradio as gr
import os
import time
import psutil
from pathlib import Path

# Processing function
def get_optimal_threads():
    cpu_count = psutil.cpu_count()
    if cpu_count <= 2:
        return 1
    elif cpu_count <= 4:
        return 2
    else:
        return cpu_count // 2
    
def upscale_image(input_image, scale_factor, model_path):
    if input_image is None:
        return None, "No image provided"
    
    try:
        # Initialize model
        if scale_factor is None:
            scale_factor = 1
        tile_size = 256 * scale_factor
        model = ESRGANOptimized(
            model_path=model_path,
            tile_size=tile_size,
            model_input_size=256,
            scale=4,
            num_threads=get_optimal_threads(),
            overlap_size=8
        )
        
        # Get original filename without extension
        original_path = Path(input_image.name)
        base_name = original_path.stem
        
        # Create output filename with '_upscaled' suffix
        output_filename = f"{base_name}_upscaled{original_path.suffix}"
        input_path = f"/tmp/{original_path.name}"
        output_path = f"/tmp/{output_filename}"
        
        input_image.save(input_path)
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
    scales = [1, 2, 3, 4, 6, 8]
    scale_dropdown = gr.Dropdown(scales, label="Upscale Factor")
    
    model_paths = {
        "4xNomos2_realplksr_dysample_256_int8_fullyoptimized.onnx": "Fully Optimized Model",
        "4x-ClearRealityV1-fp32-opset17.onnx": "ClearReality Model"
    }
    model_dropdown = gr.Dropdown(list(model_paths.values()), label="Model")
    
    iface = gr.Interface(
        fn=lambda input_image, scale_factor, model_name: upscale_image(
            input_image, scale_factor, [k for k, v in model_paths.items() if v == model_name][0]
        ),
        inputs=[gr.Image(type="pil"), scale_dropdown, model_dropdown],
        outputs=[
            gr.Image(type="filepath", label="Upscaled Image"),
            gr.Textbox(label="Status")
        ],
        title="Image Upscaler",
        description="Upload an image, select the upscale factor, and choose the model to enhance its resolution"
    )
    
    iface.launch(
        share=True,
        server_name="0.0.0.0",
        debug=True,
        server_port=7860
    )