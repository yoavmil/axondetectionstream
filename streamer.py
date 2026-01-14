# streamer.py
# Usage:
#   python streamer.py --movie path/to/video.mp4 --addr ipc:///tmp/ab.sock
#
# Sends frames over ZeroMQ as multipart messages:
#   [header_json_bytes, frame_bytes]
# Frame payload is raw BGR uint8 (OpenCV default).

import argparse
import json
import time

import cv2
import numpy as np
import zmq


class Streamer:
    def __init__(self, movie_path: str, zmq_addr: str):
        self.movie_path = movie_path
        self.zmq_addr = zmq_addr

        self.cap = cv2.VideoCapture(self.movie_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open movie: {self.movie_path}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 0 else None  # some files report 0

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.bind(self.zmq_addr)

        self.frame_id = 0

    def close(self) -> None:
        try:
            self.cap.release()
        finally:
            self.sock.close()

    def run(self) -> None:
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break

                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8, copy=False)
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)

                header = {
                    "frame_id": self.frame_id,
                    "ts_ns": time.time_ns(),
                    "shape": list(frame.shape),  # [h, w, c]
                    "dtype": str(frame.dtype),  # "uint8"
                    "encoding": "raw_bgr",
                    "fps": self.fps,
                }

                self.sock.send_multipart(
                    [json.dumps(header).encode("utf-8"), memoryview(frame)],
                    copy=False,
                )

                self.frame_id += 1
        finally:
            self.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True, help="Movie file path")
    ap.add_argument(
        "--addr",
        required=True,
        help="ZMQ bind address, e.g. ipc:///tmp/ab.sock or tcp://*:5555",
    )
    args = ap.parse_args()

    Streamer(args.movie, args.addr).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
