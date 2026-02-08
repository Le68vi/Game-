from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib 
import cv2
import winsound
import os

freq = 2500  # Frequency for the alert sound
duration = 1000  # Duration for the alert sound

def eyeAspectRatio(eye):#The eyeAspectRatio function calculates the EAR using the distances between specific eye landmarks.
    # Calculate the vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Calculate the horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)  # Calculate EAR
    return ear

count = 0
earThresh = 0.3  # Threshold for EAR
earFrames = 48  # Consecutive frames for eye closure
shapePredictor = "shape_predictor_68_face_landmarks.dat"

cam = cv2.VideoCapture(0)  # Start video capture
detector = dlib.get_frontal_face_detector()  # Initialize face detector
predictor = dlib.shape_predictor(shapePredictor)  # Initialize shape predictor

if not os.path.exists(shapePredictor):
    print("Shape predictor file not found!")
    exit()

# Get the coordinates of the left and right eye
(Istart, IEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    ret, frame = cam.read()  # Read a frame from the webcam
    if not ret:
        break

    frame = imutils.resize(frame, width=800)  # Resize the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    rects = detector(gray, 0)  # Detect faces
    for rect in rects:
        shape = predictor(gray, rect)  # Get facial landmarks
        shape = face_utils.shape_to_np(shape)

        lefteye = shape[Istart:IEnd]  # Get left eye coordinates
        righteye = shape[rstart:rEnd]  # Get right eye coordinates

        leftEar = eyeAspectRatio(lefteye)  # Calculate EAR for left eye
        rightEar = eyeAspectRatio(righteye)  # Calculate EAR for right eye

        ear = (leftEar + rightEar) / 2.0  # Average EAR

        # Draw contours around the eyes
        lefteyeHull = cv2.convexHull(lefteye)
        righteyeHull = cv2.convexHull(righteye)
        cv2.drawContours(frame, [lefteyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [righteyeHull], -1, (0, 0, 255), 1)

        if ear < earThresh:  # Check if EAR is below threshold
            count += 1
            if count >= earFrames:  # Check if drowsiness is detected
                cv2.putText(frame, "Drowsiness Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(freq, duration)  # Alert sound
        else:
            count = 0  # Reset count if eyes are open

        cv2.imshow("Frame", frame)  # Show the frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Exit on 'q' key
            break

cam.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows


"""
Eye Aspect Ratio (EAR): This is a measure used to determine whether a person's eyes are open or closed. It is calculated using the distances between specific eye landmarks.
Facial Landmark Detection: The application utilizes the Dlib library to detect facial landmarks, which helps in identifying the position of the eyes.
Thresholds: The application defines thresholds for EAR to determine when the eyes are considered closed, and it counts consecutive frames to confirm drowsiness.
"""