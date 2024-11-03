#!/usr/bin/env python3

# Import statements
from model import ESRGANOptimized
import gradio as gr
import os
import time
import psutil

def get_optimal_threads():
    cpu_count = psutil.cpu_count()
    if cpu_count <= 2:
        return 1
    elif cpu_count <= 4:
        return 2
    else:
        return cpu_count - 2
    
# Processing function
def upscale_image(input_image):
    if input_image is None:
        return None, "No image provided"
    
    try:
        # Initialize model
        model = ESRGANOptimized(
            model_path='real_esrgan_int8.onnx',
            tile_size=256,
            model_input_size=128,
            scale=4,
            num_threads=get_optimal_threads(),
            overlap_size=16
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
    iface = gr.Interface(
        fn=upscale_image,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(type="filepath", label="Upscaled Image"),
            gr.Textbox(label="Status")
        ],
        title="ESRGAN Image Upscaler",
        description="Upload an image to enhance its resolution"
    )
    
    iface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )