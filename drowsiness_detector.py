import numpy as np
import imutils
import time
import timeit
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
import make_train_data as mtd
import light_remover as lr
import ringing_alarm as alarm
import ear as er
import pygame
from check_cam_fps import check_fps 

def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESH = 0
open_ear, close_ear = er.start()
EAR_THRESH=close_ear+((open_ear-close_ear)/2)


pygame.mixer.init()
time_without_face=0
time_face_detected=0
alert_sound=pygame.mixer.Sound('alert_sound.wav')


OPEN_EAR = 0

EAR_CONSEC_FRAMES = 20 
COUNTER = 0 #Frames counter.

closed_eyes_time = [] 
TIMER_FLAG = False 
ALARM_FLAG = False 

ALARM_COUNT = 0 
RUNNING_TIME = 0 
    
PREV_TERM = 0 
#6. make trained data 
np.random.seed(9)
power, nomal, short = mtd.start(25) 
#The array the actual test data is placed.
test_data = []
#The array the actual labeld data of test data is placed.
result_data = []
#For calculate fps
prev_time = 0

#7. 
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#8.
print("starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

#####################################################################################################################

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)
    
    L, gray = lr.light_removing(frame)
    
    rects = detector(gray,0)
    
    #checking fps. If you want to check fps, just uncomment below two lines.
    #prev_time, fps = check_fps(prev_time)
    #cv2.putText(frame, "fps : {:.2f}".format(fps), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
    if len(rects)>0:
        alert_sound.stop()
        time_face_detected=time.time()
    else:
        time_without_face=time.time()-time_face_detected
        if time_without_face>=5:
            alert_sound.play()
        
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #(leftEAR + rightEAR) / 2 => both_ear. 
        both_ear = (leftEAR + rightEAR)/2  #I multiplied by 1000 to enlarge the scope.

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if both_ear < EAR_THRESH :
            if not TIMER_FLAG:
                start_closing = timeit.default_timer()
                TIMER_FLAG = True
            COUNTER += 1

            if COUNTER >= EAR_CONSEC_FRAMES:

                mid_closing = timeit.default_timer()
                closing_time = round((mid_closing-start_closing),3)

                if closing_time >= RUNNING_TIME:
                    if RUNNING_TIME == 0 :
                        CUR_TERM = timeit.default_timer()
                        OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM),3)
                        PREV_TERM = CUR_TERM
                        RUNNING_TIME = 1.75

                    RUNNING_TIME += 2
                    ALARM_FLAG = True
                    ALARM_COUNT += 1

                    print("{0}st ALARM".format(ALARM_COUNT))
                    print("The time eyes is being opened before the alarm went off :", OPENED_EYES_TIME)
                    print("closing time :", closing_time)
                    test_data.append([OPENED_EYES_TIME, round(closing_time*10,3)])
                    result = mtd.run([OPENED_EYES_TIME, closing_time*10], power, nomal, short)
                    result_data.append(result)
                    t = Thread(target = alarm.select_alarm, args = (result, ))
                    t.deamon = True
                    t.start()

        else :
            COUNTER = 0
            TIMER_FLAG = False
            RUNNING_TIME = 0

            if ALARM_FLAG :
                end_closing = timeit.default_timer()
                closed_eyes_time.append(round((end_closing-start_closing),3))
                print("The time eyes were being offed :", closed_eyes_time)

            ALARM_FLAG = False

        cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()


