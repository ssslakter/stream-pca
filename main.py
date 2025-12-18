import argparse
from stream_pca import VideoWrapper
from stream_pca import OnlineStochasticRPCA, CompressiveModel, FullRobustPCA, OnlineMovingWindowRPCA, OnlineMovingWindowRPCA_CP
from stream_pca.utils import get_device


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("filename", help="Input video file path")
    p.add_argument("--width", "-w", type=int, default=640, help="Processing width")
    p.add_argument("--ratio", "-r", type=float, default=0.1, help="Lambda parameter")
    p.add_argument("--cpu", action="store_true", help="Force CPU usage")
    p.add_argument("--prefix", "-p", type=str, default=None, help="Output prefix")
    a = p.parse_args()

    print(f"Loading {a.filename} [Width: {a.width}, Lambda: {a.ratio}]")
    dev = "cpu" if a.cpu else get_device()
    # model = CompressiveModel(subsample=a.ratio, device=dev)
    # model = FullRobustPCA(device=dev)
    # model = OnlineStochasticRPCA(device=dev)
    model = OnlineMovingWindowRPCA(device=dev)
    # model = OnlineMovingWindowRPCA_CP(device=dev)
    try:
        import time
        start_time = time.perf_counter()
        wrp = VideoWrapper(a.filename, width=a.width, max_frame=1000)
        wrp.process(model)
        end_time = time.perf_counter()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
    except KeyboardInterrupt:
        print("\nInterrupted! Saving...")
    except FileNotFoundError as e:
        return print(f"Error: {e}")

    base = a.prefix or a.filename.rsplit(".", 1)[0]
    wrp.save(f"{base}_bg.mp4", f"{base}_fg.mp4")


if __name__ == "__main__":
    main()