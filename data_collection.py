import cv2
import math
import time
import os
import numpy as np
from ultralytics import YOLO


model = YOLO("ISL.pt")

# Parameters
offset = 20
imgSize = 300
counter = 0
folder = "ISL Images"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Camera not found")
        break

    results = model(img)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf > 0.5: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Prepare white background
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            
            x1c = max(0, x1 - offset)
            y1c = max(0, y1 - offset)
            x2c = min(img.shape[1], x2 + offset)
            y2c = min(img.shape[0], y2 + offset)

            imgCrop = img[y1c:y2c, x1c:x2c]

            if imgCrop.size != 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)

                key = cv2.waitKey(1)
                if key == ord("s"):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{counter}.jpg', imgWhite)
                    print(f"Saved: {counter}")

    cv2.imshow('YOLO Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
