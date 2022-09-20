# import the necessary packages
from picamera2 import Picamera2
#from picamera2.encoders import H264Encoder
#from imutils.video import VideoStream
from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
from math import hypot
import time
import cv2
import numpy as np
import dlib
import playsound
#import imutils
import argparse

def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = image.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
        
    return gaze_ratio

# define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold for to set off the
    # alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

font = cv2.FONT_HERSHEY_PLAIN

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .wav file")

# initialize the camera and grab a reference to the raw camera capture
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
# allow the camera to warmup
time.sleep(0.1)
picam2.start()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# capture frames from the camera
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image - this array
    # will be 3D, representing the width, height, and # of channels
while True:
    image = picam2.capture_array()
    #image=np.asarray(frame.array)
    #new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    
    for face in faces:

        landmarks = predictor(gray, face)
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        #if blinking_ratio > 5.7:
            #cv2.putText(image, "BLINKING", (50, 150), font, 7, (255, 0, 0))
            
            #make sound to alert the driver
            #beep.beep(sound= 3)
        if ear < EYE_AR_THRESH:
            COUNTER += 1
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True
                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background
                    if ALARM_ON == True:
                        t = Thread(target=sound_alarm,
                                   args=("alarm.wav",))
                        t.daemon = True
                        t.start()


                    # draw an alarm on the frame
                cv2.putText(image, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False

        # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
        cv2.putText(image, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        # Gaze detection
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        if gaze_ratio <= 1:
            cv2.putText(image, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            #new_frame[:] = (0, 0, 255)
        elif 1 < gaze_ratio < 1.25:
            cv2.putText(image, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            #new_frame[:] = (255, 0, 0)
            cv2.putText(image, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
    
    # show the frame
    cv2.imshow("Frame", image)
   # cv2.imshow("new_frame", new_frame)

    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    #rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()