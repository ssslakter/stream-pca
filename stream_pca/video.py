from .model import CompressiveModel
from .model_online_dynamic import OnlineMovingWindowRPCA

import cv2, numpy as np, torch


class VideoWrapper:
    "Handles IO: Reading, Resizing, Batching, and Saving."

    def __init__(self, fn, width=160, max_frame = None):
        self.fn, self.width = fn, width
        self.cap = cv2.VideoCapture(fn)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Missing {fn}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        w_orig = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * (width / w_orig))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_frame = min(max_frame or self.n_frames, self.n_frames)        
        
        # Pre-allocate storage
        self.L_store = np.zeros((self.n_frames, self.h, self.width, 3), dtype=np.uint8)
        self.S_store = np.zeros((self.n_frames, self.h, self.width, 3), dtype=np.uint8)

    def process(self, model: CompressiveModel):
        # 1. Warmup Model
        if model is OnlineMovingWindowRPCA:
            burn_in_frames = []
            for _ in range(model.n_burnin): # Read only the burn-in length
                ret, f = self.cap.read()
                if not ret: break
                burn_in_frames.append(f) # TODO: preprocess
            
            model.init_basis(burn_in_frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, model.n_burnin)
            t = model.n_burnin
        else: 
            frames = []
            for _ in range(self.max_frame):
                ret, f = self.cap.read()
                if not ret:
                    break
                frames.append(
                    cv2.cvtColor(cv2.resize(f, (self.width, self.h)), cv2.COLOR_BGR2RGB) # h, w, 3
                )
                # print("the frame number frames", frames[-1].shape)
            model.init_basis(frames)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            t = 0

        # 2. Main Loop
        
        print(f"Processing {self.n_frames} frames...")

        while True:
            ret, f = self.cap.read()
            if not ret:
                break

            # Prep input
            f_sm = cv2.resize(f, (self.width, self.h))
            ten = (
                torch.tensor(
                    cv2.cvtColor(f_sm, cv2.COLOR_BGR2RGB),
                    dtype=torch.float32,
                    device=model.device,
                )
                / 255.0
            )

            # Inference
            bg, fg = model(ten)
            # print(fg)
            # raise ValueError
            # Store (GPU -> CPU -> UInt8)
            self.L_store[t] = (bg.cpu().numpy() * 255).astype(np.uint8)
            self.S_store[t] = (fg.cpu().numpy() * 255).astype(np.uint8) * 4

            t += 1
            if t % 50 == 0:
                print(f"Frame {t}/{self.n_frames}", end="\r")

        self.cap.release()
        return self

    def save(self, bg_fn="bg.mp4", fg_fn="fg.mp4"):
        for data, fn in zip([self.L_store, self.S_store], [bg_fn, fg_fn]):
            # Trim trailing zeros
            n = len(data)
            for i in range(n - 1, -1, -1):
                if data[i].max() > 0:
                    n = i + 1
                    break
            if n == 0:
                print(f"Skipping {fn} (empty)")
                continue

            # Setup Writer (MP4 with AVI fallback)
            fourcc, ext = cv2.VideoWriter_fourcc(*"mp4v"), "mp4"
            out = cv2.VideoWriter(fn, fourcc, self.fps, (self.width, self.h))
            if not out.isOpened():
                print(f"Falling back to MJPG for {fn}")
                fn = fn.rsplit(".", 1)[0] + ".avi"
                out = cv2.VideoWriter(
                    fn, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (self.width, self.h)
                )

            for f in data[:n]:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"Saved {fn}")
