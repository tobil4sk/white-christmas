import os
import time
import cv2
import numpy as np
import logging
from ultralytics import YOLO
import torch
from threading import Thread
import atexit
import face_recognition

DOWNSCALE_HEIGHT = 720
ENABLE_GPU = True
LOGGING = False
BLOCK_ALL = False

class VideoGetter:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW, params=[cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        (self.success, self.frame) = self.stream.read()
        self.idx = 0
        self.running = True

    def start(self):
        self.thread = Thread(target=self.get, args=())
        self.thread.daemon = True
        self.thread.start()
        return self
    
    def get(self):
        while self.running:
            if not self.success:
                self.running = False
            else:
                self.success, self.frame = self.stream.read()
                self.idx += 1
        self.stream.release()
    
    def stop(self):
        self.running = False

class FaceRecogniser:
    def __init__(self, video_getter):
        self.video_getter = video_getter
        self.known_embeddings = []
        self.face_locations = []
        self.face_midpoints: list[tuple[int, int]] = []
        self.face_markers: list[bool] = []

        for filename in os.listdir("faces"):
            path = os.path.join("faces", filename)
            # checking if it is a file
            if os.path.isfile(path):
                img = face_recognition.load_image_file(path)
                for encoding in face_recognition.face_encodings(img):
                    self.known_embeddings.append(encoding)
        self.last_frame = 0
        self.last_rec = 0

        self.running = True
    
    def get_face_midpoints(self, face_locations):
        return [((top + bottom) // 2, (right + left) // 2) for top, right, bottom, left in face_locations]

    def start(self):
        self.thread = Thread(target=self.get, args=())
        self.thread.daemon = True
        self.thread.start()
        self.thread2 = Thread(target=self.recognise, args=())
        self.thread2.daemon = True
        self.thread2.start()
        return self
    
    def recognise(self):
        while self.running:
            if self.last_rec >= self.last_frame:
                time.sleep(0.01)
                continue
            self.last_rec = self.last_frame
            small_frame = cv2.resize(self.video_getter.frame, (0, 0), fx=0.25, fy=0.25)
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = self.face_locations
            if len(face_locations) == 0:
                time.sleep(0.05)
                continue
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            self.face_markers = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_embeddings, face_encoding)
                self.face_markers.append(any(matches))
            time.sleep(1.5)

    def get(self):
        while self.running:
            if self.last_frame >= self.video_getter.idx:
                time.sleep(0.01)
                continue
            small_frame = cv2.resize(self.video_getter.frame, (0, 0), fx=0.25, fy=0.25)
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            self.last_frame = self.video_getter.idx

            self.face_locations = face_recognition.face_locations(small_frame)

            old_face_midpoints = self.face_midpoints
            old_face_markers = self.face_markers

            self.face_midpoints = self.get_face_midpoints(self.face_locations)
            self.face_markers = []

            for face_midpoint in self.face_midpoints:
                if len(old_face_midpoints) == 0:
                    break
                distances = np.linalg.vector_norm(face_midpoint - np.array(old_face_midpoints))
                i = np.argmin(distances)

                old_face_midpoints.pop(i)
                if len(old_face_markers) == 0:
                    self.face_markers.append(False)
                else:
                    self.face_markers.append(old_face_markers.pop(i))

    def stop(self):
        self.running = False


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
print("Loaded Video Stream")

# print("Loading Face Detector")
# face_detector = FaceDetector(video_getter).start()
# print("Loaded Face Detector")

print("Loading Face Recogniser")
face_recogniser = FaceRecogniser(video_getter).start()
print("Loaded Face Recogniser")

def cleanup():
    face_recogniser.stop()
    video_getter.stop()

atexit.register(cleanup)

# Accepts arguments of the form [xMin, yMin, xMax, yMax]
def contains_box(a, b):
    # Check if box a contains box b
    a_contains_b = (a[0] <= b[0] and a[1] <= b[1] and
                    a[2] >= b[2] and a[3] >= b[3])

    # Check if box b contains box a
    b_contains_a = (b[0] <= a[0] and b[1] <= a[1] and
                    b[2] >= a[2] and b[3] >= a[3])

    # Return True if either box contains the other
    return a_contains_b or b_contains_a

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

    face_locations = list(face_recogniser.face_locations)
    face_markers = list(face_recogniser.face_markers)
    
    perf_log("Segment")

    merged_mask = np.ones((downscale_size[1], downscale_size[0]), dtype=np.uint8)
    
    outlines = np.full((frame_size[1], frame_size[0]), 0, dtype=np.uint8)
    for result in results:
        if result.masks is None:
            break
        for mask, box in zip(result.masks, result.boxes.xyxy):
            isblocked = False
            i = -1
            for (top, right, bottom, left), blocked in zip(face_locations, face_markers):
                i += 1
                if not blocked:
                    continue
                if contains_box(box.numpy() / downscale_factor, [left * 4, bottom * 4, right * 4, top * 4]):
                    isblocked = True
                    face_locations.pop(i)
                    face_markers.pop(i)
                    break
            if not isblocked and not BLOCK_ALL:
                continue
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

    # for result in results:
    #     if result.boxes is None:
    #         break
    #     for box in result.boxes.xyxy:
    #         left, bottom, right, top = box.numpy() / downscale_factor
    #         cv2.rectangle(final, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)

    # face_locations = face_recogniser.face_locations
    # face_markers = face_recogniser.face_markers
    # for (top, right, bottom, left), blocked in zip(face_locations, face_markers):
    #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #     top *= 4
    #     right *= 4
    #     bottom *= 4
    #     left *= 4

    #     # Draw a box around the face
    #     cv2.rectangle(final, (left, top), (right, bottom), (0, 0, 255), 2)

    #     # Draw a label with a name below the face
    #     cv2.rectangle(final, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     cv2.putText(final, "blocked" if blocked else "allowed", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
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