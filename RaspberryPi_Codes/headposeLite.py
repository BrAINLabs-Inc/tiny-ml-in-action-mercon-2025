import cv2
import numpy as np
import time
import tensorflow as tf
from collections import deque

def _load_tflite_interpreter(tflite_path: str, num_threads: int = 1) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter

def _quantize_input(arr_f32: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return arr_f32.astype(np.float32)
    scale, zero = input_detail["quantization"]  # (scale, zero_point)
    if scale == 0:
        return arr_f32.astype(dtype)
    q = np.round(arr_f32 / scale + zero)
    if dtype == np.uint8:
        q = np.clip(q, 0, 255)
    elif dtype == np.int8:
        q = np.clip(q, -128, 127)
    return q.astype(dtype)

def _dequantize_output(arr, output_detail: dict) -> np.ndarray:
    dtype = output_detail["dtype"]
    if dtype == np.float32:
        return arr.astype(np.float32)
    scale, zero = output_detail["quantization"]
    if scale == 0:
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - zero) * scale

def _preprocess_bgr_to_model_input(frame_bgr, input_size=(64, 64), grayscale=True, normalize_if_float=True):
    if grayscale:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = frame_bgr
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_AREA)
    if grayscale:
        img = np.expand_dims(img, -1)  # (H,W,1)
    arr = img.astype(np.float32)
    if normalize_if_float:
        arr /= 255.0
    arr = np.expand_dims(arr, 0)  # (1,H,W,C)
    return arr

def run_tflite_headpose_camera(
    tflite_path: str,
    camera_index: int = 1,
    backend: int = cv2.CAP_AVFOUNDATION,  # macOS; Linux: cv2.CAP_V4L2; Windows: cv2.CAP_DSHOW
    input_size=(64, 64),
    grayscale=True,
    normalize_if_float=True,
    num_threads: int = 1,
    quit_key: str = 'q'
):
    """
    Opens webcam, runs TFLite head-pose inference, and overlays:
      - Pitch, Yaw, Roll (float outputs)
      - Per-frame latency (ms) and average latency
      - Smoothed FPS
    Press `quit_key` to exit.
    """
    # Load model
    interpreter = _load_tflite_interpreter(tflite_path, num_threads=num_threads)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_det = input_details[0]
    out_det = output_details[0]

    # Open camera
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam opened successfully.")
    print(f"Press '{quit_key}' to exit.")

    # Warm-up (allocate kernels, avoid first-frame spike)
    dummy = np.zeros((1, input_size[0], input_size[1], 1 if grayscale else 3), dtype=np.float32)
    q_dummy = _quantize_input(dummy, in_det)
    interpreter.set_tensor(in_det["index"], q_dummy)
    interpreter.invoke()

    latencies = deque(maxlen=60)
    fps_hist = deque(maxlen=60)
    last_t = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Could not read frame.")
                break

            # Preprocess -> float32 (1,H,W,C) in [0,1]
            inp_f32 = _preprocess_bgr_to_model_input(
                frame, input_size=input_size, grayscale=grayscale, normalize_if_float=normalize_if_float
            )

            # Resize interpreter input if needed
            expected_shape = tuple(in_det["shape"])
            if inp_f32.shape != expected_shape:
                interpreter.resize_tensor_input(in_det["index"], inp_f32.shape)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                in_det = input_details[0]
                out_det = output_details[0]

            # Quantize if model expects int8/uint8
            q_inp = _quantize_input(inp_f32, in_det)

            # Inference + latency
            interpreter.set_tensor(in_det["index"], q_inp)
            t0 = time.perf_counter()
            interpreter.invoke()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            # Output (dequantized to float32)
            raw_out = interpreter.get_tensor(out_det["index"])
            preds = _dequantize_output(raw_out, out_det)
            pitch, yaw, roll = preds[0]

            # FPS
            now = time.perf_counter()
            fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now
            fps_hist.append(fps)
            avg_fps = sum(fps_hist) / len(fps_hist)
            avg_lat = sum(latencies) / len(latencies)

            # Overlay on the original color frame
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw:   {yaw:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll:  {roll:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(frame, f"Latency: {latency_ms:.1f} ms (avg {avg_lat:.1f} ms)",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Head Pose Estimation (TFLite)", frame)

            if cv2.waitKey(1) & 0xFF == ord(quit_key):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

run_tflite_headpose_camera(
    tflite_path="pruned_quantized_model.tflite",
    camera_index=1,                   # adjust to pick your built-in webcam
    backend=cv2.CAP_AVFOUNDATION,     # macOS
    input_size=(64, 64),
    grayscale=True,
    normalize_if_float=True,
    num_threads=2
)
