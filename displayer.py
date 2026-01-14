# displayer.py
# Usage:
#   python displayer.py --in tcp://127.0.0.1:5556 --blur
#   python displayer.py --in tcp://127.0.0.1:5556 --blur --blur-ksize 31
# Message format from Detector:
#   [header_json_bytes, frame_bytes, contours_json_bytes]
# header must include: shape, dtype, ts_ms
#
# Display logic:
# 1) Block until a message arrives
# 2) Read ts_ms from payload
# 3) Sleep until it's time to show it (relative to first received frame)
# 4) Display frame with contours + ts text
# Repeat

import argparse
import json
import time

import cv2
import numpy as np
import zmq


class Displayer:
    def __init__(self, in_addr: str, blur_inside_contours: bool, blur_ksize: int):
        self.ctx = zmq.Context.instance()
        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.connect(in_addr)

        self.stream_t0_ms = 0.0
        self.wall_t0 = None
        self.inited = False

        self.blur_inside_contours = blur_inside_contours
        # must be odd and >= 3 for GaussianBlur
        if blur_ksize < 3:
            blur_ksize = 3
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        self.blur_ksize = blur_ksize

    @staticmethod
    def _contours_to_cv(contours_payload):
        out = []
        for poly in contours_payload:
            out.append(np.array(poly, dtype=np.int32).reshape(-1, 1, 2))
        return out

    @staticmethod
    def _format_ts_ms(ts_ms):
        if ts_ms is None:
            return "t=NA"
        return f"t={ts_ms:.0f} ms"

    def _apply_blur_inside(self, img_bgr: np.ndarray, contours_cv) -> np.ndarray:
        if not contours_cv:
            return img_bgr

        # Create mask of contour interiors
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours_cv, -1, 255, thickness=-1)

        # Blur entire image, then composite only where mask==255
        blurred = cv2.GaussianBlur(img_bgr, (self.blur_ksize, self.blur_ksize), 0)

        out = img_bgr.copy()
        out[mask == 255] = blurred[mask == 255]
        return out

    def run(self) -> None:
        cv2.namedWindow("Displayer", cv2.WINDOW_NORMAL)

        try:
            while True:
                # 1) block until ZMQ ready (a message arrives)
                parts = self.pull.recv_multipart()

                if len(parts) != 3:
                    continue

                header = json.loads(parts[0].decode("utf-8"))

                # Optional EOS support (won't block at end if you use it)
                if header.get("eos"):
                    break

                frame_bytes = parts[1]
                contours_payload = json.loads(parts[2].decode("utf-8"))

                # 2) read timestamp from payload
                ts_ms = header.get("ts_ms")
                if ts_ms is None:
                    ts_ms = 0.0
                ts_ms = float(ts_ms)

                if not self.inited:
                    # Use first received frame as time origin
                    self.stream_t0_ms = ts_ms
                    self.wall_t0 = time.perf_counter()
                    self.inited = True

                # 3) sleep until the time for the image arrives
                target_wall = self.wall_t0 + (ts_ms - self.stream_t0_ms) / 1000.0
                now = time.perf_counter()
                sleep_s = target_wall - now
                if sleep_s > 0:
                    time.sleep(sleep_s)

                # Decode frame
                shape = header["shape"]
                dtype = np.dtype(header["dtype"])
                frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape).copy()

                contours_cv = self._contours_to_cv(contours_payload)

                if self.blur_inside_contours:
                    vis = self._apply_blur_inside(
                        frame, contours_cv
                    )  # returns writable copy
                else:
                    vis = frame

                # Draw contour outlines on top (optional but usually desired)
                if contours_cv:
                    cv2.drawContours(vis, contours_cv, -1, (0, 255, 0), 2)

                cv2.putText(
                    vis,
                    self._format_ts_ms(ts_ms),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Displayer", vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        finally:
            self.pull.close()
            cv2.destroyAllWindows()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_addr", required=True, help="ZMQ input address (B -> C)"
    )
    ap.add_argument("--blur", action="store_true", help="Blur inside detected contours")
    ap.add_argument(
        "--blur-ksize", type=int, default=31, help="Gaussian blur kernel size (odd int)"
    )
    args = ap.parse_args()

    Displayer(
        args.in_addr, blur_inside_contours=args.blur, blur_ksize=args.blur_ksize
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
