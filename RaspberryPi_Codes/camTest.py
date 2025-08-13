import cv2

# Try indices 0..5 and show which one actually gives frames
for idx in range(6):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)  # macOS backend
    ok, frame = cap.read()
    print(f"Index {idx}: {'OK' if ok else 'No frame'}")
    if ok:
        cv2.imshow(f"Camera {idx}", frame)
        cv2.waitKey(500)  # brief peek
        cv2.destroyAllWindows()
    cap.release()