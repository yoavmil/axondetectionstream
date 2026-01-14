# detector.py
# Usage:
#   python detector.py --in tcp://127.0.0.1:5555 --out tcp://*:5556

import argparse
import json

import cv2
import imutils
import numpy as np
import zmq


class Detector:
    def __init__(self, in_addr: str, out_addr: str):
        self.ctx = zmq.Context.instance()

        # A -> B
        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.connect(in_addr)

        # B -> C
        self.push = self.ctx.socket(zmq.PUSH)
        self.push.bind(out_addr)

        self.prev_frame = None
        self.counter = 0

    def run(self) -> None:
        while True:
            header_bytes, frame_bytes = self.pull.recv_multipart()

            header = json.loads(header_bytes.decode("utf-8"))
            shape = header["shape"]  # [h, w, c]
            dtype = np.dtype(header["dtype"])

            frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            contours_payload = []

            if self.counter == 0:
                self.prev_frame = gray_frame
            else:
                diff = cv2.absdiff(gray_frame, self.prev_frame)
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)

                cnts = cv2.findContours(
                    thresh.copy(),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cnts = imutils.grab_contours(cnts)

                # serialize contours as lists
                contours_payload = [cnt.reshape(-1, 2).tolist() for cnt in cnts]

                self.prev_frame = gray_frame

            self.counter += 1

            out_header = {
                "frame_id": header["frame_id"],
                "ts_ns": header["ts_ns"],
                "fps": header.get("fps"),
                "shape": shape,
                "dtype": str(dtype),
                "encoding": "raw_bgr",
            }

            self.push.send_multipart(
                [
                    json.dumps(out_header).encode("utf-8"),
                    memoryview(frame),
                    json.dumps(contours_payload).encode("utf-8"),
                ],
                copy=False,
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_addr", required=True, help="ZMQ input address (A -> B)"
    )
    ap.add_argument(
        "--out", dest="out_addr", required=True, help="ZMQ output address (B -> C)"
    )
    args = ap.parse_args()

    Detector(args.in_addr, args.out_addr).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
