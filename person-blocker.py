import time
import cv2
import numpy as np
import logging
from ultralytics import YOLO
import torch
from threading import Thread
import atexit

class VideoGetter:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src, cv2.CAP_ANY, params=[cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        (self.success, self.frame) = self.stream.read()
        self.running = True

    def start(self):
        self.thread = Thread(target=self.get, args=())
        self.thread.start()
        return self
    
    def get(self):
        while self.running:
            if not self.success:
                self.stop()
            else:
                self.success, self.frame = self.stream.read()
        self.stream.release()
    
    def stop(self):
        self.running = False

DOWNSCALE_HEIGHT = 720
ENABLE_GPU = True
LOGGING = True

try: # GPU Support?
    torch.cuda.set_device(0)
    torch.cuda.is_available = lambda : ENABLE_GPU
except:
    pass

print("Loading Models")
seg_model = YOLO("yolo11n-seg.pt")
print("Loaded Segmentation Model")

logging.getLogger('ultralytics').setLevel(logging.ERROR)

frames = 0
last_time = time.perf_counter()

perf_timestamp = time.perf_counter()
def perf_log(label):
    global perf_timestamp
    timestamp = time.perf_counter()
    if LOGGING:
        print(label + ":", round((timestamp - perf_timestamp) * 1000, 2))
    perf_timestamp = timestamp

print("Loading Video Stream")
video_getter = VideoGetter(0).start()
atexit.register(video_getter.stop)
print("Loaded Video Stream")

# loop through frame
while video_getter.running:
    # Break loop outcome 
    if cv2.waitKey(1) & 0xFF == ord('q') or not video_getter.running:
        video_getter.stop()
        break

    if LOGGING:
        perf_timestamp = time.perf_counter()
        print("\n=== NEW FRAME ===\n")

    frame = video_getter.frame
    
    perf_log("Grab Frame")

    frame_size = (frame.shape[1], frame.shape[0])

    downscale_factor =  DOWNSCALE_HEIGHT / frame_size[0]

    downscale_size = (int(frame_size[0] * downscale_factor), int(frame_size[1] * downscale_factor))
    downscaled = cv2.resize(frame, downscale_size, interpolation=cv2.INTER_CUBIC)

    perf_log("Downscale")

    # YOLO detection
    results = seg_model(downscaled, classes=[0])
    
    perf_log("Segment")

    merged_mask = np.ones((downscale_size[1], downscale_size[0]), dtype=np.uint8)
    
    outlines = np.full((frame_size[1], frame_size[0]), 0, dtype=np.uint8)

    for result in results:
        if result.masks is None:
            break
        for mask in result.masks:
            points = np.int32([mask.xy])
            cv2.fillPoly(merged_mask, points, 0)
            cv2.polylines(outlines, np.rint(points / downscale_factor).astype(np.int32), True, 200, 2)

    scaled_mask = cv2.resize(merged_mask, frame_size)
    final = cv2.bitwise_and(frame, frame, mask=scaled_mask)

    perf_log("Mask+Outlines")

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

    perf_log("Effect")
    
    masked_effect = cv2.bitwise_and(effect, effect, mask=(1 - scaled_mask))
    masked_effect -= outlines
    masked_effect[masked_effect > (255 - outlines)] = 0
    masked_effect = cv2.blur(masked_effect, (3, 3))

    final += cv2.cvtColor(masked_effect, cv2.COLOR_GRAY2BGR)
    
    perf_log("Apply Effect + Outlines")
    
    frames += 1
    curr_time = time.perf_counter()
    if curr_time - last_time > 1:
        print("FPS:", frames)
        last_time = curr_time
        frames = 0

    # Show result to user on desktop
    cv2.imshow('Output', final)
    
    perf_log("Display")

cv2.destroyAllWindows() # Closes imshow frames