#!/usr/bin/env python3

# Import statements
from model import ESRGANOptimized
import gradio as gr
import os
import time
import psutil

# Processing function
def get_optimal_threads():
    cpu_count = psutil.cpu_count()
    if cpu_count <= 2:
        return 1
    elif cpu_count <= 4:
        return 2
    else:
        return cpu_count // 2
    
def upscale_image(input_image, scale_factor):
    if input_image is None:
        return None, "No image provided"
    
    try:
        # Initialize model
        if scale_factor is None:
            scale_factor = 1
        tile_size = 256 * scale_factor
        model = ESRGANOptimized(
            model_path='4xNomos2_realplksr_dysample_256_int8_fullyoptimized.onnx',
            tile_size=tile_size,
            model_input_size=256,
            scale=4,
            num_threads=get_optimal_threads(),
            overlap_size=8
        )
        
        # Save and process image
        input_path = "/tmp/input.png"
        output_path = "/tmp/output.png"
        
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
    
    iface = gr.Interface(
        fn=lambda input_image, scale_factor: upscale_image(input_image, scale_factor),
        inputs=[gr.Image(type="pil"), scale_dropdown],
        outputs=[
            gr.Image(type="filepath", label="Upscaled Image"),
            gr.Textbox(label="Status")
        ],
        title="Image Upscaler",
        description="Upload an image and select the upscale factor to enhance its resolution"
    )
    
    iface.launch(
        share=True,
        server_name="0.0.0.0",
        debug=True,
        server_port=7860
    )