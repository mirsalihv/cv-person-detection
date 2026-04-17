import cv2
import torch
import torchvision
import torchvision.transforms as T
import time


# ===== MODEL =====
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===== LOAD MODEL =====
model = get_model()
model.load_state_dict(torch.load("models/model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

print("Model loaded")


# ===== CAMERA =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

transform = T.ToTensor()

# ===== FULLSCREEN =====
cv2.namedWindow("Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ===== CALIBRATION =====
focal_length = 800  # adjust for real accuracy


# ===== VARIABLES =====
start_time = time.time()
last_update_time = time.time()

window_seconds = 20

close_count = 0
total_count = 0
percent = 0

best_height = 0
best_frame = None

# tracking
next_id = 0
tracked_objects = {}


# ===== HELPERS =====
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# ===== MAIN LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1920, 1080))

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = T.ToTensor()(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])

    boxes = outputs[0]["boxes"].cpu()
    scores = outputs[0]["scores"].cpu()

    new_tracked = {}

    for box, score in zip(boxes, scores):
        if score > 0.5:
            box = list(map(int, box))
            x1, y1, x2, y2 = box

            pixel_height = y2 - y1

            # ===== DISTANCE =====
            if pixel_height > 0:
                distance_m = (focal_length * 1.7) / pixel_height
            else:
                distance_m = 0

            if distance_m < 1:
                label = "VERY CLOSE"
            elif distance_m < 2:
                label = "CLOSE"
            else:
                label = "FAR"

            # ===== TRACKING =====
            cx, cy = get_center(box)
            assigned_id = None

            for obj_id, prev_box in tracked_objects.items():
                px, py = get_center(prev_box)
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

                if dist < 60:
                    assigned_id = obj_id
                    break

            if assigned_id is None:
                assigned_id = next_id
                next_id += 1

                total_count += 1
                if label in ["CLOSE", "VERY CLOSE"]:
                    close_count += 1

            new_tracked[assigned_id] = box

            # ===== BEST FRAME =====
            if pixel_height > best_height:
                best_height = pixel_height
                best_frame = frame.copy()

            # ===== DRAW =====
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.putText(frame, f"ID {assigned_id}", (x1, y1-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            cv2.putText(frame, f"{distance_m:.2f} m", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    tracked_objects = new_tracked


    # ===== 1-SECOND LIVE UPDATE =====
    current_time = time.time()

    if current_time - last_update_time > 1:
        if total_count > 0:
            percent = (close_count / total_count) * 100
        else:
            percent = 0

        print(f"[LIVE] {percent:.2f}% close")
        last_update_time = current_time


    # ===== 20-SECOND WINDOW =====
    if current_time - start_time > window_seconds:

        print(f"[FINAL 20s] {percent:.2f}% came close")

        if best_frame is not None and best_height > 100:
            filename = f"BEST_{int(time.time())}.jpg"
            cv2.imwrite(filename, best_frame)
            print("Saved:", filename)

        # reset
        start_time = current_time
        close_count = 0
        total_count = 0
        best_height = 0
        best_frame = None


    # ===== UI =====
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.putText(frame, f"People: {len(tracked_objects)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Close %: {percent:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) == 27:
        break


# ===== CLEANUP =====
cap.release()
cv2.destroyAllWindows()