import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from stream_pca import (
    VideoWrapper, 
    OnlineStochasticRPCA, 
    CompressiveModel, 
    FullRobustPCA, 
    OnlineMovingWindowRPCA
)

def calculate_sparsity(S_tensor, epsilon=0.01):
    """Percentage of non-zero (foreground) elements."""
    non_zero = torch.count_nonzero(S_tensor > epsilon)
    return (non_zero.item() / S_tensor.numel()) * 100

def calculate_nuclear_norm(L_tensor):
    """Nuclear norm as a proxy for low-rankness."""
    # Reshape to (Pixels, Channels) to treat as a matrix for singular value analysis
    flat_L = L_tensor.view(-1, 3)
    return torch.linalg.norm(flat_L, ord='nuc').item()

def run_benchmark(video_path, width=320, max_frames=2000):
    models = {
        "Full RPCA (ADMM)": FullRobustPCA(),
        "Stochastic Online": OnlineStochasticRPCA(rank=50),
        "Moving Window": OnlineMovingWindowRPCA(n_win=30, n_burnin=60, rank=50),
        "Compressive": CompressiveModel(subsample=0.1)
    }

    results = {}

    for name, model in models.items():
        print(f"\n>>> Benchmarking: {name}")
        wrp = VideoWrapper(video_path, width=width, max_frame=max_frames)
        
        # Track metrics
        sparsity_log = []
        rank_proxy_log = []
        
        start_time = time.time()
        
        # Custom process loop to extract metrics per frame
        # (Similar to your VideoWrapper.process but with logging)
        if hasattr(model, 'n_burnin'):
            # Handle Moving Window Burn-in
            frames = []
            for _ in range(model.n_burnin):
                ret, f = wrp.cap.read()
                if not ret: break
                frames.append(cv2.cvtColor(cv2.resize(f, (wrp.width, wrp.h)), cv2.COLOR_BGR2RGB))
            model.init_basis(frames)
        else:
            # Standard Init
            frames = []
            wrp.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(min(50, max_frames)):
                ret, f = wrp.cap.read()
                if not ret: break
                frames.append(cv2.cvtColor(cv2.resize(f, (wrp.width, wrp.h)), cv2.COLOR_BGR2RGB))
            model.init_basis(frames)

        wrp.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        t = 0
        while t < max_frames:
            ret, f = wrp.cap.read()
            if not ret: break
            
            f_sm = cv2.resize(f, (wrp.width, wrp.h))
            ten = torch.tensor(cv2.cvtColor(f_sm, cv2.COLOR_BGR2RGB), 
                               dtype=torch.float32, device=model.device) / 255.0
            
            L, S = model(ten)
            
            sparsity_log.append(calculate_sparsity(S))
            rank_proxy_log.append(calculate_nuclear_norm(L))
            t += 1

        end_time = time.time()
        
        results[name] = {
            "time": end_time - start_time,
            "sparsity": sparsity_log,
            "rank": rank_proxy_log,
            "fps": t / (end_time - start_time)
        }

    return results

def plot_results(results):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # 1. Execution Time (Bar Chart)
    names = list(results.keys())
    times = [results[n]['time'] for n in names]
    ax1.bar(names, times, color=['blue', 'green', 'red', 'orange'])
    ax1.set_title("Total Processing Time (Lower is Better)")
    ax1.set_ylabel("Seconds")

    # 2. Sparsity Level (Line Chart)
    for name in names:
        ax2.plot(results[name]['sparsity'], label=f"{name} (Mean: {np.mean(results[name]['sparsity']):.2f}%)")
    ax2.set_title("Sparsity Level of S (Foreground) over Time")
    ax2.set_ylabel("% Non-zero Pixels")
    ax2.legend()

    # 3. Low-Rankness / Nuclear Norm (Line Chart)
    for name in names:
        # Normalize by mean to see variance in background stability
        data = np.array(results[name]['rank'])
        ax3.plot(data / np.max(data), label=name)
    ax3.set_title("Normalized Nuclear Norm of L (Background Stability)")
    ax3.set_ylabel("Relative Nuclear Norm")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    plt.show()

if __name__ == "__main__":
    # Replace with your video file path
    VIDEO_FILE = "station.mp4" 
    bench_data = run_benchmark(VIDEO_FILE)
    plot_results(bench_data)