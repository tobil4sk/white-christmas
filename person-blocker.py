import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2
import numpy as np

# load model (once)
bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16
))

# get prediction result
# get vid cap device
USE_REMOTE = False
if not USE_REMOTE:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("http://10.253.169.237:8080/video") 
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

threshold = 0.55
TARGET_HEIGHT = 1000

DILATE_RADIUS = 20
DILATE_ELEMENT = (DILATE_RADIUS * 2 + 1, DILATE_RADIUS * 2 + 1)  # Make kernel size (2*size + 1)
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_ELEMENT)

# loop through frame
while cap.isOpened():
    ret, frame = cap.read()
    frame_size = (frame.shape[1], frame.shape[0])

    downscale_factor =  TARGET_HEIGHT / frame_size[0]

    downscale_size = (int(frame_size[0] * downscale_factor), int(frame_size[1] * downscale_factor))
    downscaled = cv2.resize(frame, downscale_size, interpolation=cv2.INTER_CUBIC)

    # BodyPix Detections
    result = bodypix_model.predict_single(downscaled)
    mask = 1 - result.get_mask(threshold=threshold).numpy().astype(np.uint8)
    if (cv2.waitKey(10) & 0xFF == ord('r')):
        mask = cv2.dilate(mask, DILATE_KERNEL)

    scaled_mask = np.expand_dims(cv2.resize(mask, frame_size, interpolation=cv2.INTER_CUBIC), axis=-1)
    masked_image = cv2.bitwise_and(frame, frame, mask=scaled_mask)

    final = masked_image.astype(np.int16)

    lowres_size = (frame_size[0] // 20, frame_size[1] // 20)
    lowres_effect = (255 - cv2.cvtColor(cv2.resize(frame, lowres_size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)).astype(np.int16)
    lowres_noise = np.random.randint(-15, 15, (lowres_size[1], lowres_size[0]), dtype=np.int16)
    lowres_effect += 60 + lowres_noise
    lowres_effect = np.clip(lowres_effect, 120, 255).astype(np.uint8)
    effect = cv2.blur(cv2.resize(lowres_effect, frame_size, interpolation=cv2.INTER_NEAREST), (5, 5))
    noise = np.random.randint(0, 25, (frame_size[1] // 2, frame_size[0] // 2), dtype=np.int16)
    effect = np.clip(effect.astype(np.int16) + cv2.resize(noise, frame_size, interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
    masked_effect = cv2.bitwise_and(effect, effect, mask=(1 - scaled_mask))

    final += cv2.cvtColor(masked_effect, cv2.COLOR_GRAY2BGR)

    outline = cv2.Canny(scaled_mask, 0, 1)
    outline = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)
    outline = cv2.blur(outline, (3, 3))

    final -= outline
    final = np.clip(final, 0, 255)
    
    # Show result to user on desktop
    cv2.imshow('Output', final.astype(np.uint8))
    
    # Break loop outcome 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release() # Releases webcam or capture device
cv2.destroyAllWindows() # Closes imshow frames