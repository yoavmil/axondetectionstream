# displayer.py
# Usage:
#   python displayer.py --in tcp://127.0.0.1:5556 --fps 25
#
# Receives from Detector (B) as multipart:
#   [header_json_bytes, frame_bytes, contours_json_bytes]
# Displays frames paced to the requested FPS, draws contours, and shows the timestamp.

import argparse
import json
import time
from collections import deque

import cv2
import numpy as np
import zmq


class Displayer:
    def __init__(self, in_addr: str, fps: float, buffer_size: int = 60):
        self.in_addr = in_addr
        self.fps = float(fps)
        self.period_s = 1.0 / self.fps
        self.buffer = deque(maxlen=buffer_size)

        self.ctx = zmq.Context.instance()
        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.connect(self.in_addr)

        # Non-blocking receive so display timing controls the loop
        self.pull.setsockopt(zmq.RCVTIMEO, 0)

    def _recv_available(self) -> None:
        while True:
            try:
                parts = self.pull.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return

            if len(parts) != 3:
                continue  # ignore malformed

            header = json.loads(parts[0].decode("utf-8"))
            frame_bytes = parts[1]
            contours = json.loads(parts[2].decode("utf-8"))

            shape = header["shape"]
            dtype = np.dtype(header["dtype"])
            frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

            ts_ns = header.get("ts_ns")  # may or may not exist depending on B
            self.buffer.append((frame.copy(), contours, ts_ns))

    @staticmethod
    def _contours_to_cv(contours_payload):
        cv_contours = []
        for poly in contours_payload:
            # poly: [[x,y], [x,y], ...]
            arr = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv_contours.append(arr)
        return cv_contours

    @staticmethod
    def _format_ts(ts_ns):
        if ts_ns is None:
            return "ts: N/A"
        # show as wall-clock time with milliseconds
        sec = ts_ns / 1e9
        lt = time.localtime(sec)
        ms = int((sec - int(sec)) * 1000)
        return time.strftime("%H:%M:%S", lt) + f".{ms:03d}"

    def run(self) -> None:
        cv2.namedWindow("Displayer", cv2.WINDOW_NORMAL)

        next_tick = time.perf_counter()

        while True:
            # keep pulling frames into buffer as fast as possible
            self._recv_available()

            # Pace display
            now = time.perf_counter()
            sleep_s = next_tick - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            next_tick += self.period_s

            if self.buffer:
                frame, contours_payload, ts_ns = self.buffer.popleft()
                vis = frame.copy()

                cv_contours = self._contours_to_cv(contours_payload)
                if cv_contours:
                    cv2.drawContours(vis, cv_contours, -1, (0, 255, 0), 2)

                cv2.putText(
                    vis,
                    self._format_ts(ts_ns),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Displayer", vis)

            # handle quit
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        self.pull.close()
        cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_addr", required=True, help="ZMQ input address (B -> C)")
    ap.add_argument("--fps", type=float, required=True, help="Original FPS to display at (e.g. 25)")
    ap.add_argument("--buffer", type=int, default=60, help="Frame buffer size (frames)")
    args = ap.parse_args()

    Displayer(args.in_addr, args.fps, buffer_size=args.buffer).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
