#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2 as cv
import os
import torch
from ultralytics import YOLOWorld
from paddleocr import PaddleOCR
from rapidfuzz import fuzz

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str,
                        help="Path to input video file. If not provided, the camera will be used.")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (0 is default webcam).")
    parser.add_argument("--width", type=int, default=1080, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--items", type=str, default="person,phone",
                        help="Comma-separated list of items to search for in OCR text.")
    parser.add_argument("--yolo_model", type=str, default="yolov8s-world.pt",
                        help="Path or name of the YOLOWorld model weights.")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Confidence threshold for YOLO detection.")
    return parser.parse_args()

def main():
    args = get_args()

    # 1. Parse desired items into a list.
    target_items = [item.strip().lower() for item in args.items.split(",") if item.strip()]

    # 2. Open video or webcam.
    if args.video:
        cap = cv.VideoCapture(args.video)
        input_ext = os.path.splitext(args.video)[1].lower()
        if input_ext not in VIDEO_EXTENSIONS:
            input_ext = ".mp4"
    else:
        cap = cv.VideoCapture(args.device)
        input_ext = ".mp4"

    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # 3. Prepare output writer.
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    base = (os.path.splitext(os.path.basename(args.video))[0]
            if args.video else "camera_output")
    output_path = os.path.join(results_folder, f"{base}_processed{input_ext}")
    fourcc = cv.VideoWriter_fourcc(*('mp4v' if input_ext=='.mp4' else 'XVID'))
    fps = cap.get(cv.CAP_PROP_FPS) or 25
    frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    out_writer = cv.VideoWriter(output_path, fourcc, fps, frame_size)

    print(f"Source FPS: {fps}, Frame size: {frame_size}")
    print(f"Saving to: {output_path}")

    # 4. Initialize PaddleOCR on CPU
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en')

    # 5. Initialize YOLOWorld on Metal (mps) or CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLOWorld(args.yolo_model).to(device)
    print(f"Using YOLOWorld model: {args.yolo_model} on {device}")

    # 6. Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of input.")
            break

        # YOLOWorld inference
        results = model.predict(frame, conf=args.conf_threshold, verbose=False)
        det_boxes = results[0].boxes if results else []

        overlay = frame.copy()

        # For each detection, crop & OCR
        for box in det_boxes:
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < args.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = model.names.get(cls_idx, "unknown")

            roi = frame[y1:y2, x1:x2]
            ocr_res = ocr.ocr(roi)

            texts = []
            if ocr_res and ocr_res[0]:
                for line in ocr_res[0]:
                    texts.append(line[1][0])
            recognized = " ".join(texts).lower()

            # fuzzy match via rapidfuzz
            match = None
            best_score = 0
            for item in target_items:
                score = fuzz.partial_ratio(item, recognized)
                if score > best_score and score >= 80:
                    best_score, match = score, item

            color = (0,255,0) if match else (0,0,255)
            cv.rectangle(overlay, (x1,y1), (x2,y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            if match:
                label += f" | {match}:{best_score}"
            cv.putText(overlay, label, (x1, max(y1-10,0)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if recognized:
                cv.putText(overlay, f"OCR: {recognized}", (x1, y2+20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        cv.imshow("YOLOWorld + OCR", overlay)
        out_writer.write(overlay)

        if cv.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    out_writer.release()
    cv.destroyAllWindows()
    print("Done. Output saved to:", output_path)


if __name__ == "__main__":
    main()
