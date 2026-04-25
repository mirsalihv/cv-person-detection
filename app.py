import cv2
import time
import math

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

focal_length = 800
real_face_height = 0.16  # meters

tracked_faces = {}
next_id = 0

start_time = time.time()
last_update_time = time.time()
window_seconds = 20

close_count = 0
total_count = 0
percent = 0

best_height = 0
best_frame = None

def get_center(x, y, w, h):
    return (x + w // 2, y + h // 2)

# main loops
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    new_tracked = {}

    for (x, y, w, h) in faces:

        # distance meter
        if h > 0:
            distance = (focal_length * real_face_height) / h
        else:
            distance = 0

        if distance < 1:
            label = "VERY CLOSE"
        elif distance < 2:
            label = "CLOSE"
        else:
            label = "FAR"

        # tracking
        cx, cy = get_center(x, y, w, h)
        assigned_id = None

        for face_id, (px, py, pw, ph) in tracked_faces.items():
            pcx, pcy = get_center(px, py, pw, ph)
            dist = math.sqrt((cx - pcx)**2 + (cy - pcy)**2)

            if dist < 60:
                assigned_id = face_id
                break

        if assigned_id is None:
            assigned_id = next_id
            next_id += 1

            total_count += 1
            if label in ["CLOSE", "VERY CLOSE"]:
                close_count += 1

        new_tracked[assigned_id] = (x, y, w, h)

        # best shots
        if h > best_height:
            best_height = h
            best_frame = frame.copy()
        ##
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)

        cv2.putText(frame, f"ID {assigned_id}", (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.putText(frame, f"{distance:.2f} m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

    tracked_faces = new_tracked

    # 1 sec upd
    current_time = time.time()

    if current_time - last_update_time > 1:
        percent = (close_count / total_count) * 100 if total_count > 0 else 0
        print(f"[LIVE] {percent:.2f}% close")
        last_update_time = current_time

    # 20 sec upd
    if current_time - start_time > window_seconds:
        print(f"[FINAL 20s] {percent:.2f}% came close")

        if best_frame is not None and best_height > 50:
            filename = f"BEST_{int(time.time())}.jpg"
            cv2.imwrite(filename, best_frame)
            print("Saved:", filename)

        start_time = current_time
        close_count = 0
        total_count = 0
        best_height = 0
        best_frame = None

    # UI
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.putText(frame, f"Faces: {len(tracked_faces)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Close %: {percent:.1f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Face System (OpenCV)", frame)

    if cv2.waitKey(1) == 27:
        break



cap.release()
cv2.destroyAllWindows()