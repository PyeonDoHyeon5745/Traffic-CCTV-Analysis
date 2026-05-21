from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "best.pt"
VIDEO_PATH = "testvideo.mp4"
DATA_YAML  = "data.yaml"

model = YOLO(MODEL_PATH)

# ── 1. mAP50 ──────────────────────────────────────────────
print("=" * 40)
print("[1] mAP50 측정 중 (val)...")
metrics = model.val(data=DATA_YAML, imgsz=960, verbose=False)

map50     = metrics.box.map50
map50_95  = metrics.box.map

print(f"  mAP50      : {map50:.4f}  ({map50*100:.2f}%)")
print(f"  mAP50-95   : {map50_95:.4f}  ({map50_95*100:.2f}%)")

# ── 2. FPS ────────────────────────────────────────────────
print()
print("[2] FPS 측정 중 (영상 첫 200프레임 샘플링)...")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("  영상 열기 실패")
else:
    WARMUP   = 5    # 워밍업 프레임 수 (제외)
    SAMPLE   = 200  # 측정할 프레임 수

    frame_idx = 0
    elapsed   = 0.0

    while frame_idx < WARMUP + SAMPLE:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < WARMUP:
            # 워밍업 - GPU/메모리 초기화
            model(frame, imgsz=960, verbose=False)
            frame_idx += 1
            continue

        t0 = time.perf_counter()
        model(frame, imgsz=960, verbose=False)
        elapsed += time.perf_counter() - t0
        frame_idx += 1

    measured = frame_idx - WARMUP
    if measured > 0 and elapsed > 0:
        fps = measured / elapsed
        print(f"  측정 프레임  : {measured}")
        print(f"  총 소요 시간 : {elapsed:.2f}s")
        print(f"  FPS          : {fps:.2f}")
    else:
        print("  FPS 측정 실패")

    cap.release()

# ── 요약 ──────────────────────────────────────────────────
print()
print("=" * 40)
print("[ 최종 결과 ]")
print(f"  mAP50  : {map50*100:.2f}%")
if measured > 0 and elapsed > 0:
    print(f"  FPS    : {fps:.2f}")
print("=" * 40)
