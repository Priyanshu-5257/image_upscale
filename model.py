import os
import sys
import numpy as np
from PIL import Image, ImageFilter
import onnxruntime
import time
import gc
from threading import Thread
from queue import Queue
import psutil


class ESRGANOptimized:
    def __init__(self, model_path, tile_size=256, model_input_size=128, prepad=0, scale=4, num_threads=2, overlap_size=32):
        self.model_path = model_path
        self.base_tile_size = tile_size  # Base size for splitting the image
        self.model_input_size = model_input_size  # Fixed model input size (128)
        self.model_output_size = model_input_size * scale  # Model output size (512)
        self.prepad = prepad
        self.scale = scale
        self.overlap_size = overlap_size  # Size of overlap between tiles
        self.num_threads = min(num_threads, psutil.cpu_count() or 2)
        self._init_model()
        self.tile_queue = Queue(maxsize=4)
        self.result_queue = Queue()

    def _init_model(self):
        self.session_opti = onnxruntime.SessionOptions()
        self.session_opti.enable_mem_pattern = False
        self.session_opti.intra_op_num_threads = 1
        self.session_opti.inter_op_num_threads = 1
        self.session_opti.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self.session_opti.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = onnxruntime.InferenceSession(
            self.model_path, 
            self.session_opti, 
            providers=['CPUExecutionProvider']
        )
        self.model_input = self.session.get_inputs()[0].name

    def _calculate_tile_dimensions(self, img_width, img_height):
        """Calculate optimal tile dimensions based on image size and model constraints."""
        # Ensure tile size is divisible by model input size
        tile_size = self.base_tile_size - (self.base_tile_size % self.model_input_size)
        print(tile_size)
        # Calculate number of tiles needed
        num_width = int(np.ceil((img_width + self.overlap_size) / (tile_size - self.overlap_size)))
        num_height = int(np.ceil((img_height + self.overlap_size) / (tile_size - self.overlap_size)))
        print(num_width, num_height)
        return tile_size, num_width, num_height

    def _create_blending_mask(self, tile_size):
        """Create a blending mask for smooth tile transitions."""
        mask = np.ones((tile_size, tile_size), dtype=np.float32)
        
        # Create feathered edges for blending
        fade_dist = self.overlap_size
        for i in range(fade_dist):
            # Apply cosine interpolation for smooth blending
            fade = 0.5 * (1 - np.cos(np.pi * i / fade_dist))
            
            # Fade all edges
            mask[i, :] *= fade  # Top edge
            mask[-i-1, :] *= fade  # Bottom edge
            mask[:, i] *= fade  # Left edge
            mask[:, -i-1] *= fade  # Right edge
            
        return mask

    def _tile_preprocess(self, img):
        """Preprocess tile for model input."""
        # Resize tile to model input size
        img = img.resize((self.model_input_size, self.model_input_size), Image.BILINEAR)
        #img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        input_data = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return input_data[np.newaxis, ...]

    def _process_tile_worker(self):
        """Worker thread for processing individual tiles."""
        while True:
            item = self.tile_queue.get()
            if item is None:
                self.tile_queue.task_done()
                break

            idx, tile, original_size = item
            try:
                # Process the tile
                result = self.session.run([], {self.model_input: tile})[0][0]
                result = np.clip(result.transpose(1, 2, 0), 0, 1) * 255.0
                result_img = Image.fromarray(result.round().astype(np.uint8))
                
                # Resize to match the scaled original size
                if result_img.size != original_size:
                    result_img = result_img.resize(original_size, Image.LANCZOS)
                
                del result
                gc.collect()
                
                self.result_queue.put((idx, result_img))
            except Exception as e:
                print(f"Error processing tile {idx}: {str(e)}")
                self.result_queue.put((idx, None))
            
            self.tile_queue.task_done()

    def process_image(self, image_path, output_path):
        try:
            img = Image.open(image_path).convert('RGB')
            self.width, self.height = img.size
            
            # Calculate optimal tile dimensions
            tile_size, self.num_width, self.num_height = self._calculate_tile_dimensions(self.width, self.height)
            total_tiles = self.num_width * self.num_height
            
            # Create blending mask
            blend_mask = self._create_blending_mask(tile_size)
            blend_mask = np.stack([blend_mask] * 3, axis=-1)  # Create 3-channel mask for RGB
            
            # Calculate padded dimensions
            pad_width = (self.num_width - 1) * (tile_size - self.overlap_size) + tile_size
            pad_height = (self.num_height - 1) * (tile_size - self.overlap_size) + tile_size
            
            # Create padded image
            pad_img = Image.new("RGB", (pad_width, pad_height))
            pad_img.paste(img)

            # Start worker threads
            threads = []
            for _ in range(self.num_threads):
                t = Thread(target=self._process_tile_worker)
                t.daemon = True
                t.start()
                threads.append(t)

            # Process tiles
            print(f"Processing {total_tiles} tiles...")
            for i in range(self.num_height):
                for j in range(self.num_width):
                    idx = i * self.num_width + j
                    
                    # Calculate tile position with overlap
                    x_start = j * (tile_size - self.overlap_size)
                    y_start = i * (tile_size - self.overlap_size)
                    
                    # Extract tile
                    box = [
                        x_start,
                        y_start,
                        min(x_start + tile_size, pad_width),
                        min(y_start + tile_size, pad_height)
                    ]
                    
                    tile = pad_img.crop(box)
                    processed = self._tile_preprocess(tile)
                    
                    # Calculate the expected output size for this tile
                    output_size = (
                        (box[2] - box[0]) * self.scale,
                        (box[3] - box[1]) * self.scale
                    )
                    
                    self.tile_queue.put((idx, processed, output_size))
                    
                    del tile
                    del processed
                    
                    if idx % 5 == 0:
                        gc.collect()

            # Signal workers to finish
            for _ in range(self.num_threads):
                self.tile_queue.put(None)

            # Create output image with scaled dimensions
            out_width = self.width * self.scale
            out_height = self.height * self.scale
            out_img = np.zeros((out_height, out_width, 3), dtype=np.float32)
            out_blend = np.zeros((out_height, out_width, 3), dtype=np.float32)
            
            # Scale up the blending mask
            scaled_tile_size = tile_size * self.scale
            scaled_overlap = self.overlap_size * self.scale
            scaled_blend_mask = np.array(Image.fromarray(
                (blend_mask * 255).astype(np.uint8)
            ).resize((scaled_tile_size, scaled_tile_size), Image.LANCZOS)) / 255.0

            # Collect and blend tiles
            for _ in range(total_tiles):
                idx, result_tile = self.result_queue.get()
                if result_tile is None:
                    raise Exception(f"Failed to process tile {idx}")
                
                # Calculate position
                i = idx // self.num_width
                j = idx % self.num_width
                
                x_start = j * (scaled_tile_size - scaled_overlap)
                y_start = i * (scaled_tile_size - scaled_overlap)
                
                # Convert tile to numpy array
                tile_array = np.array(result_tile, dtype=np.float32)
                
                # Calculate the valid region for this tile
                h, w = tile_array.shape[:2]
                y_end = min(y_start + h, out_height)
                x_end = min(x_start + w, out_width)
                
                # Adjust blend mask size if needed
                blend_h, blend_w = y_end - y_start, x_end - x_start
                curr_blend_mask = scaled_blend_mask[:blend_h, :blend_w]
                
                # Add tile to output using the blending mask
                out_img[y_start:y_end, x_start:x_end] += tile_array[:blend_h, :blend_w] * curr_blend_mask
                out_blend[y_start:y_end, x_start:x_end] += curr_blend_mask
                
                del result_tile
                del tile_array
                if idx % 5 == 0:
                    gc.collect()

            # Normalize the blended image
            out_blend = np.maximum(out_blend, 1e-6)
            out_img = out_img / out_blend
            out_img = np.clip(out_img, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image and save
            Image.fromarray(out_img).save(output_path, "JPEG", optimize=True)
            
            # Wait for threads to finish
            for t in threads:
                t.join()
            
            del out_img
            del out_blend
            gc.collect()
            
            return True

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return False

def get_optimal_threads():
    cpu_count = psutil.cpu_count()
    if cpu_count <= 2:
        return 1
    elif cpu_count <= 4:
        return 2
    else:
        return cpu_count // 2

def main():
    model_path = '4xNomos2_realplksr_dysample_256_int8_fullyoptimized.onnx'
    input_path = sys.argv[1] if len(sys.argv) > 1 else "/home/priyanshu/Downloads/images/unedited-photos-bee-on-sunflower-600nw-2474014217.jpg"
    output_path = f'{input_path}_upscaled.png'
    
    print('Initializing ESRGAN...')
    model = ESRGANOptimized(
        model_path=model_path,
        tile_size=512,  # Base size for splitting the image
        model_input_size=256,  # Fixed model input size
        scale=4,
        num_threads=get_optimal_threads(),
        overlap_size=8  # Size of overlap between tiles
    )
    
    start_time = time.time()
    success = model.process_image(input_path, output_path)
    
    if success:
        print(f'Result saved to: {output_path}')
        print(f'Time cost: {time.time()-start_time:.4f}s')
    else:
        print("Failed to process image")

if __name__ == "__main__":
    main()