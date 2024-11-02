import cv2
import numpy as np
import logging
from ultralytics import YOLO

# get prediction result
# get vid cap device
USE_REMOTE = False
if not USE_REMOTE:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("http://10.253.169.237:8080/video") 
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

seg_model = YOLO("yolo11n-seg.pt")

threshold = 0.55
TARGET_HEIGHT = 1000

DILATE_RADIUS = 20
DILATE_ELEMENT = (DILATE_RADIUS * 2 + 1, DILATE_RADIUS * 2 + 1)  # Make kernel size (2*size + 1)
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_ELEMENT)

logging.getLogger('ultralytics').setLevel(logging.ERROR)

# loop through frame
while cap.isOpened():
    ret, frame = cap.read()
    frame_size = (frame.shape[1], frame.shape[0])

    downscale_factor =  TARGET_HEIGHT / frame_size[0]

    downscale_size = (int(frame_size[0] * downscale_factor), int(frame_size[1] * downscale_factor))
    downscaled = cv2.resize(frame, downscale_size, interpolation=cv2.INTER_CUBIC)

    # YOLO detection
    results = seg_model(downscaled)

    merged_mask = np.ones((downscale_size[1], downscale_size[0]), dtype=np.uint8)
    
    outlines = np.zeros((frame_size[1], frame_size[0], 1), dtype=np.uint8)

    for result in results:
        for mask, box in zip(result.masks, result.boxes):
            name = seg_model.names[int(box.cls)]
            if name != 'person':
                continue
            points = np.int32([mask.xy])
            cv2.fillPoly(merged_mask, points, 0)
            cv2.polylines(outlines, (points / downscale_factor).astype(np.int32), True, 255, 1)

    scaled_mask = cv2.resize(merged_mask, frame_size)
    
    final = cv2.bitwise_and(frame, frame, mask=scaled_mask)

    final = final.astype(np.int16)

    lowres_size = (frame_size[0] // 15, frame_size[1] // 15)
    lowres_effect = (255 - cv2.cvtColor(cv2.resize(frame, lowres_size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)).astype(np.int16)
    lowres_noise = np.random.randint(-15, 15, (lowres_size[1], lowres_size[0]), dtype=np.int16)
    lowres_effect += 60 + lowres_noise
    lowres_effect = np.clip(lowres_effect, 120, 255).astype(np.uint8)
    effect = cv2.blur(cv2.resize(lowres_effect, frame_size, interpolation=cv2.INTER_NEAREST), (5, 5))
    noise = np.random.randint(0, 25, (frame_size[1] // 2, frame_size[0] // 2), dtype=np.int16)
    effect = np.clip(effect.astype(np.int16) + cv2.resize(noise, frame_size, interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
    masked_effect = cv2.bitwise_and(effect, effect, mask=(1 - scaled_mask))

    outlines = cv2.cvtColor(outlines, cv2.COLOR_GRAY2BGR)
    outlines = cv2.blur(outlines, (3, 3))

    final = np.clip(final - outlines, 0, 255)

    final += cv2.cvtColor(masked_effect, cv2.COLOR_GRAY2BGR)

    final = np.clip(final, 0, 255).astype(np.uint8)

    # Show result to user on desktop
    cv2.imshow('Output', final)
    
    # Break loop outcome 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release() # Releases webcam or capture device
cv2.destroyAllWindows() # Closes imshow frames