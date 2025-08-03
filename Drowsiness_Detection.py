from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

# Initialize mixer for alarm sound
mixer.init()
mixer.music.load("alarm.wav")  # Make sure this sound file is present

# Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # vertical
    B = distance.euclidean(mouth[4], mouth[8])   # vertical
    C = distance.euclidean(mouth[0], mouth[6])   # horizontal
    mar = (A + B) / (2.0 * C)
    return mar

# Thresholds
EYE_THRESH = 0.25
MOUTH_THRESH = 0.75
FRAME_CHECK = 20  # Number of consecutive frames
YAWN_FRAME_CHECK = 15  # For long yawn detection

# Dlib face & shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Indexes for landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Camera
cap = cv2.VideoCapture(0)
eye_flag = 0
yawn_flag = 0
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Eye and mouth landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # Compute EAR and MAR
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Draw landmarks
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)

        # Eye closure detection
        if ear < EYE_THRESH:
            eye_flag += 1
            if eye_flag >= FRAME_CHECK and not alarm_on:
                mixer.music.play(-1)
                alarm_on = True
            if eye_flag >= FRAME_CHECK:
                cv2.putText(frame, "********** ALERT: DROWSY EYES **********", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            eye_flag = 0

        # Yawning detection
        if mar > MOUTH_THRESH:
            yawn_flag += 1
            if yawn_flag >= YAWN_FRAME_CHECK and not alarm_on:
                mixer.music.play(-1)
                alarm_on = True
            if yawn_flag >= YAWN_FRAME_CHECK:
                cv2.putText(frame, "********** ALERT: YAWNING TOO LONG **********", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            yawn_flag = 0

        # Stop alarm when both EAR and MAR are normal
        if ear >= EYE_THRESH and mar <= MOUTH_THRESH:
            if alarm_on:
                mixer.music.stop()
                alarm_on = False

    # Display
    cv2.imshow("Driver Drowsiness + Yawning Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
