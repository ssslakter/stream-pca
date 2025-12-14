import argparse
from stream_pca import CompressiveModel, VideoWrapper


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("filename", help="Input video file path")
    p.add_argument("--width", "-w", type=int, default=640, help="Processing width (resizes input)")
    p.add_argument("--ratio", "-r", type=float, default=0.1, help="Subsample ratio (0.0 to 1.0)")
    p.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    p.add_argument("--prefix", "-p", type=str, default=None, help="Output filename prefix")
    a = p.parse_args()

    print(f"Loading {a.filename} [Width: {a.width}, Sample Ratio: {a.ratio}]")
    dev = "cpu" if a.cpu else None
    model = CompressiveModel(subsample=a.ratio, device=dev)

    try:
        wrp = VideoWrapper(a.filename, width=a.width)
        wrp.process(model)
    except KeyboardInterrupt:
        print("\nInterrupted by user! Saving current progress...")
    except FileNotFoundError as e:
        return print(f"Error: {e}")

    base = a.prefix or a.filename.rsplit(".", 1)[0]
    wrp.save(f"{base}_bg.mp4", f"{base}_fg.mp4")


if __name__ == "__main__":
    main()
