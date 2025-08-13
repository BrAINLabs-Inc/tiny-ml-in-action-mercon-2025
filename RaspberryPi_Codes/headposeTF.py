import cv2
import numpy as np
import time
from collections import deque

from headposeModel import build_mobilenet_simplified

def run_headpose_camera(
    model_weights_path='best_model_weights.h5',
    camera_index=1,
    backend=cv2.CAP_AVFOUNDATION,   # macOS; on Linux, try cv2.CAP_V4L2; on Windows, cv2.CAP_DSHOW
    input_size=(64, 64),
    normalize=True,                 # set False if your model expects 0..255 scale
    dense_units=(512, 256, 128),
    learning_rate=0.01,
    num_outputs=3,
    quit_key='q'
):
    """
    Opens a webcam stream, runs head pose prediction per frame, and overlays:
      - Pitch, Yaw, Roll
      - Per-frame prediction latency (ms)
      - Smoothed FPS

    Press the `quit_key` to exit.
    """

    # ----- Build and load model -----
    model = build_mobilenet_simplified(
        input_shape=(input_size[0], input_size[1], 1),
        num_outputs=num_outputs,
        dense_units=dense_units,
        learning_rate=learning_rate,
        compile_model=True
    )
    model.load_weights(model_weights_path)

    # ----- Open camera -----
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully.")
    print(f"Press '{quit_key}' to exit.")

    # Warm-up: run one dummy prediction to initialize kernels/graph, avoiding first-frame latency spike
    dummy = np.zeros((1, input_size[0], input_size[1], 1), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    # Stats
    latencies_ms = deque(maxlen=60)  # rolling window of last 60 frames
    fps_times = deque(maxlen=60)
    last_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Preprocess to grayscale (HxW -> (H,W,1) -> (1,H,W,1))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, input_size, interpolation=cv2.INTER_AREA)
            inp = resized.astype('float32')
            if normalize:
                inp = inp / 255.0
            inp = np.expand_dims(inp, axis=-1)   # (H,W,1)
            inp = np.expand_dims(inp, axis=0)    # (1,H,W,1)

            # Predict + measure latency
            t0 = time.perf_counter()
            preds = model.predict(inp, verbose=0)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency_ms)

            pitch, yaw, roll = preds[0]

            # FPS calculation (smoothed)
            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps_times.append(fps)
            avg_fps = sum(fps_times) / len(fps_times)
            avg_lat = sum(latencies_ms) / len(latencies_ms)

            # Overlay text
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw:   {yaw:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll:  {roll:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Latency: {latency_ms:.1f} ms  (avg {avg_lat:.1f} ms)",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Head Pose Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()



run_headpose_camera(
    model_weights_path='best_model_weights.h5',
    camera_index=1,                  # change if needed
    backend=cv2.CAP_AVFOUNDATION,    # macOS
    input_size=(64, 64),
    normalize=True
)
