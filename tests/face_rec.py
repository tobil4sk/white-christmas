import face_recognition
import cv2
import numpy as np
import numpy.typing as npt
import os
import sys

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

if len(sys.argv) > 1:
    video_source = sys.argv[1]
else:
    # Get a reference to webcam #0 (the default one)
    video_source = 0

video_capture = cv2.VideoCapture(video_source)

# # Load a sample picture and learn how to recognize it.
# obama_image = face_recognition.load_image_file("obama.jpg")
# obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# # Load a second sample picture and learn how to recognize it.
# biden_image = face_recognition.load_image_file("biden.jpg")
# biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# frank_image = face_recognition.load_image_file("frank.jpg")
# frank_face_encoding = face_recognition.face_encodings(frank_image)[0]

known_face_encodings = []
known_face_names = []

for filename in os.listdir("faces"):
    f = os.path.join("faces", filename)
    # checking if it is a file
    if os.path.isfile(f):
        img = face_recognition.load_image_file(f)
        known_face_encodings.append(face_recognition.face_encodings(img)[0])
        known_face_names.append(filename[:-4])




# Initialize some variables
face_locations = []
face_encodings = []
face_midpoints: list[tuple[int, int]] = []
face_markers: list[bool] = []
frame_count = 0
FRAME_BETWEEN_MATCHING = 10

def get_face_midpoints(face_locations: list[tuple[int, int, int, int]]):
    # return np.array([[(top + bottom) // 2, (right + left) // 2] for top, right, bottom, left in face_locations])
    return [((top + bottom) // 2, (right + left) // 2) for top, right, bottom, left in face_locations]

process_this_frame = True

while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame , cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        old_face_midpoints = face_midpoints
        old_face_markers = face_markers

        face_midpoints = get_face_midpoints(face_locations)
        face_markers = []

        if frame_count == 0 or len(face_midpoints) != len(old_face_midpoints):
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            frame_count = FRAME_BETWEEN_MATCHING

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_markers.append(any(matches))
        else:
            frame_count -= 1

            for face_midpoint in face_midpoints:
                distances = np.linalg.vector_norm(face_midpoint - np.array(old_face_midpoints))
                i = np.argmin(distances)

                old_face_midpoints.pop(i)
                face_markers.append(old_face_markers.pop(i))

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), blocked in zip(face_locations, face_markers):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "blocked" if blocked else "allowed", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
