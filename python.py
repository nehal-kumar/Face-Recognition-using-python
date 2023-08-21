import cv2
import face_recognition
import numpy as np

# Load images of known faces and their names
known_face_encodings = []
known_face_names = []
image_filenames = ['person1.jpg', 'person2.jpg']
for filename in image_filenames:
    image = face_recognition.load_image_file(filename)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(filename[:-4])

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find all faces in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face and see if it matches a known face
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around the face and label it
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
