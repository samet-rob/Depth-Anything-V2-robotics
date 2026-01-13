import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2

def run_depth(video_path, outdir, encoder='vitl', input_size=518):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    os.makedirs(outdir, exist_ok=True)
    
    filename = video_path
    print(f'Processing video: {filename}')
    
    raw_video = cv2.VideoCapture(filename)
    frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    total_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = os.path.join(outdir, 'results_depth.mp4')
    # Using mp4v for compatibility
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    pbar = tqdm(total=total_frames, desc="Generating Depth Map")
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        
        with torch.no_grad():
            depth = depth_anything.infer_image(raw_frame, input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # Apply colormap to get 3-channel depth map
        depth_color = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        out.write(depth_color)
        pbar.update(1)
    
    pbar.close()
    raw_video.release()
    out.release()
    print(f'Depth video saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Wrapper')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--outdir', default='../../data', help='Output directory')
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    args = parser.parse_args()
    
    run_depth(args.video, args.outdir, args.encoder)
