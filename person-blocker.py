import time
import cv2
import numpy as np
import logging
from ultralytics import YOLO
import torch

DOWNSCALE_HEIGHT = 1000
ENABLE_GPU = True

try: # GPU Support?
    torch.cuda.set_device(0)
    torch.cuda.is_available = lambda : ENABLE_GPU
except:
    pass

seg_model = YOLO("yolo11n-seg.pt")

logging.getLogger('ultralytics').setLevel(logging.ERROR)

# get vid cap device
USE_REMOTE = False
if not USE_REMOTE:
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF, params=[cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
else:
    cap = cv2.VideoCapture("http://10.253.169.237:8080/video", cv2.CAP_MSMF, params=[cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

frames = 0
last_time = time.perf_counter()

# loop through frame
while cap.isOpened():
    ret, frame = cap.read()

    frame_size = (frame.shape[1], frame.shape[0])

    downscale_factor =  DOWNSCALE_HEIGHT / frame_size[0]

    downscale_size = (int(frame_size[0] * downscale_factor), int(frame_size[1] * downscale_factor))
    downscaled = cv2.resize(frame, downscale_size, interpolation=cv2.INTER_CUBIC)

    # YOLO detection
    results = seg_model(downscaled)

    merged_mask = np.ones((downscale_size[1], downscale_size[0]), dtype=np.uint8)
    
    outlines = np.full((frame_size[1], frame_size[0]), 0, dtype=np.uint8)

    for result in results:
        if result.masks is None:
            break
        for mask, box in zip(result.masks, result.boxes):
            name = seg_model.names[int(box.cls)]
            if name != 'person':
                continue
            points = np.int32([mask.xy])
            cv2.fillPoly(merged_mask, points, 0)
            cv2.polylines(outlines, np.rint(points / downscale_factor).astype(np.int32), True, 200, 2)

    scaled_mask = cv2.resize(merged_mask, frame_size)
    final = cv2.bitwise_and(frame, frame, mask=scaled_mask)

    lowres_size = (frame_size[0] // 20, frame_size[1] // 20)
    lowres_effect = (255 - cv2.cvtColor(cv2.resize(frame, lowres_size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY))
    lowres_noise = np.random.randint(20, 50, (lowres_size[1], lowres_size[0]), dtype=np.uint8)
    lowres_effect += lowres_noise
    lowres_effect[lowres_effect < lowres_noise] = 255 # Underflow Protection
    lowres_effect[lowres_effect < 100] = 100 # Clamp low values so we don't get too dark colours
    effect = cv2.resize(lowres_effect.astype(np.uint8), frame_size, interpolation=cv2.INTER_NEAREST)
    noise = np.random.randint(0, 75, (frame_size[1] // 2, frame_size[0] // 2), dtype=np.uint8)
    noise = cv2.resize(noise, frame_size, interpolation=cv2.INTER_LINEAR)
    effect += noise
    effect[effect < noise] = 255
    
    masked_effect = cv2.bitwise_and(effect, effect, mask=(1 - scaled_mask))
    masked_effect -= outlines
    masked_effect[masked_effect > (255 - outlines)] = 0
    masked_effect = cv2.blur(masked_effect, (3, 3))

    final += cv2.cvtColor(masked_effect, cv2.COLOR_GRAY2BGR)
    
    frames += 1
    curr_time = time.perf_counter()
    if curr_time - last_time > 1:
        print("FPS:", frames)
        last_time = curr_time
        frames = 0

    # Show result to user on desktop
    cv2.imshow('Output', final)

    # Break loop outcome 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # Releases webcam or capture device
cv2.destroyAllWindows() # Closes imshow frames