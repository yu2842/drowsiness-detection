import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QInputDialog
import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import pygame
from scipy.spatial import distance
####
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5.uic import loadUi
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
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTimeEdit, QPushButton, QLabel, QMessageBox
from PyQt5.QtCore import QTimer, QTime, Qt
from PyQt5.QtMultimedia import QSound
from PyQt5.uic import loadUi

########
def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESH = 0
#open_ear, close_ear = er.start()
#EAR_THRESH=close_ear+((open_ear-close_ear)/2)
#EAR_THRESH=0.2

# 기타 설정
frames_to_measure = {"open": 90, "closed": 150}  # 눈을 뜬 상태와 감은 상태를 측정할 프레임 수
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

"""
class Earm(QWidget):
    def __init__(self):
        super().__init__()

        # 레이아웃 및 위젯 초기화
        layout = QVBoxLayout()

        self.label = QLabel("Output will be displayed here.", self)
        layout.addWidget(self.label)

        self.button = QPushButton("Start Measurement", self)
        self.button.clicked.connect(self.start_measurement)
        layout.addWidget(self.button)

        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.start_timer_app)
        layout.addWidget(self.next_button)

        # EAR 값을 저장할 클래스 멤버 변수 초기화
        self.open_ear = 0.0
        self.closed_ear = 0.0
        self.combined_ear = 0.0  # 새로운 변수 추가

        self.setLayout(layout)

    def start_measurement(self):
        global cap, detector, predictor, lStart, lEnd, rStart, rEnd

        # 클래스 멤버 변수로 EAR 값을 저장
        self.open_ear, self.closed_ear = self.measure_ear(frames_to_measure["open"], frames_to_measure["closed"])
        play_notification_sound()

        # 새로운 변수에 값을 할당
        self.combined_ear = self.closed_ear + ((self.open_ear - self.closed_ear) / 2)

        self.label.setText(
            f"눈을 뜨고 있는 상태의 EAR: {self.open_ear:.4f}\n눈을 감은 상태의 EAR: {self.closed_ear:.4f}\n"
            f"Combined EAR: {self.combined_ear:.4f}")

        # 여기서 self.open_ear, self.closed_ear, self.combined_ear를 사용하여 필요한 동작을 수행할 수 있습니다.
        print("눈을 뜨고 있는 상태의 EAR:", self.open_ear)
        print("눈을 감은 상태의 EAR:", self.closed_ear)
        print("Combined EAR:", self.combined_ear)

        # 캠 릴리스
        cap.release()
        cv2.destroyAllWindows()

        # 여기서도 self.open_ear, self.closed_ear, self.combined_ear를 사용할 수 있습니다.
        # 예를 들어, 다른 함수로 전달하거나 외부 변수에 저장할 수 있습니다.

    def measure_ear(self, open_duration, closed_duration):
        global cap, detector, predictor, lStart, lEnd, rStart, rEnd

        open_message = f"3초 동안 눈을 뜨고 있어주세요"
        closed_message = f"5초 동안 눈을 감고 있어주세요"

        open_ear = self.measure_single_ear(open_duration, open_message)
        play_notification_sound()
        closed_ear = self.measure_single_ear(closed_duration, closed_message)
        play_notification_sound()

        return open_ear, closed_ear

    def measure_single_ear(self, duration, message):
        global cap, detector, predictor, lStart, lEnd, rStart, rEnd

        QInputDialog.getText(self, "Instruction", message + "\n측정을 원한다면 OK를 입력해주세요.")

        print("측정을 시작합니다.")

        ear_list = []

        for _ in range(duration):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                face = faces[0]
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                left_eye = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]

                ear = calculate_EAR(left_eye + right_eye)
                ear_list.append(ear)

                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            cv2.imshow("Frame", frame)
            cv2.waitKey(30)

        average_ear = sum(ear_list) / len(ear_list)
        print("Average EAR:", average_ear)
        print("측정이 완료되었습니다.")

        return average_ear

    def start_timer_app(self):
        self.timer_app = TimerApp()

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.timer_app)

        self.setLayout(layout)
        self.close()
"""

def play_notification_sound():
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound('ok.wav')  # 알림음 파일명을 적절히 변경
    alert_sound.play()
    pygame.time.delay(1000)  # 1초 동안 대기

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

def start_detection():
    global alert_sound
    global EAR_THRESH
    #pygame.mixer.init()
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

class TimerApp(QWidget):
    global alert_sound
    def __init__(self):
        super().__init__()

        # 타이머, 남은 시간, 일시정지 상태에 대한 변수 초기화
        self.timer = QTimer()
        self.remaining_time = QTime(0, 0)
        self.paused = False

        # 사용자 인터페이스 초기화
        self.init_ui()

    def init_ui(self):
        # 시간을 설정할 수 있는 위젯 생성
        self.time_edit = QTimeEdit(self)
        self.time_edit.setDisplayFormat("HH:mm:ss")

        # 시작, 일시정지, 정지 버튼 생성 및 이벤트 핸들러 연결
        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_timer)
        self.start_button.clicked.connect(start_detection)

        self.pause_button = QPushButton('Pause', self)
        self.pause_button.clicked.connect(self.pause_timer)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_timer)

        # 종료 메시지를 표시할 레이블 생성
        self.end_label = QLabel('', self)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.time_edit)
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.end_label)

        self.setLayout(layout)

        # 타이머의 timeout 시그널에 대한 슬롯 연결
        self.timer.timeout.connect(self.update_timer)

        # 윈도우의 크기와 제목 설정 및 표시
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Timer App')
        self.show()

    def start_timer(self):
        # 알람 여부를 선택하는 다이얼로그 표시
        reply = QMessageBox.question(self, '알람 설정', '알람을 울릴까요?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        # 타이머가 활성화되어 있지 않은 경우에만 시작
        if not self.timer.isActive():
            self.remaining_time = self.time_edit.time()
            self.timer.start(1000)  # 타이머를 1초마다 업데이트
            self.paused = False

        # 사용자가 Yes를 선택한 경우에만 알람 설정
        if reply == QMessageBox.Yes:
            self.alarm_enabled = True
        else:
            self.alarm_enabled = False

    def pause_timer(self):
        # 타이머가 활성화되어 있는 경우에만 일시정지
        if self.timer.isActive():
            self.timer.stop()
            self.paused = True

    def stop_timer(self):
        # 타이머가 활성화되어 있는 경우에만 정지
        if self.timer.isActive():
            self.timer.stop()
            self.time_edit.setTime(QTime(0, 0))
            self.end_label.setText('End')

    def update_timer(self):
        # 일시정지 상태가 아닌 경우에만 시간을 업데이트
        if not self.paused:
            self.remaining_time = self.remaining_time.addSecs(-1)
            self.time_edit.setTime(self.remaining_time)

            # 남은 시간이 0이면 타이머 정지하고 종료 메시지 표시
            if self.remaining_time == QTime(0, 0):
                self.timer.stop()
                self.end_label.setText('End')

                # 알람이 활성화된 경우에만 알람 소리 재생
                if self.alarm_enabled:
                    alert_sound=pygame.mixer.Sound('alert_sound.wav')
                    alert_sound.play()            

class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        loadUi('main.ui', self)
        
        # main.ui에서 precautions 버튼에 클릭 이벤트를 연결
        self.precautionsbutton.clicked.connect(self.show_precautions_dialog)
        
        # main.ui에서 start 버튼에 클릭 이벤트를 연결
        #self.sbutton.clicked.connect(self.start_timer_app)
        self.sbutton.clicked.connect(self.show_setting_dialog)
        self.how.clicked.connect(self.show_how_dialog)
        
    def show_precautions_dialog(self):
        # precautions.ui를 불러와서 다이얼로그로 띄우기
        precautions_dialog = PrecautionsDialog()
        precautions_dialog.exec_()
    
    def show_setting_dialog(self):
        setting_dialog = SettingDialog()
        setting_dialog.exec_()
        
    def show_how_dialog(self):
        # precautions.ui를 불러와서 다이얼로그로 띄우기
        how_dialog = HowDialog()
        how_dialog.exec_()
    
    """
    def start_ear_m(self):
        global cl
        self.earm = Earm()
        self.earm.show()
        cl = self.earm.combined_ear
    """  
        
class PrecautionsDialog(QDialog):
    def __init__(self):
        super(PrecautionsDialog, self).__init__()
        loadUi('precautions.ui', self)

        # precautions.ui에서 home 버튼에 클릭 이벤트를 연결
        self.homebutton.clicked.connect(self.close_dialog)

    def close_dialog(self):
        # 다이얼로그를 닫기
        self.close()
        
class HowDialog(QDialog):
    def __init__(self):
        super(HowDialog, self).__init__()
        loadUi('how.ui', self)

        # precautions.ui에서 home 버튼에 클릭 이벤트를 연결
        self.homebutton.clicked.connect(self.close_dialog)

    def close_dialog(self):
        # 다이얼로그를 닫기
        self.close()

class TiDialog(QDialog):
    def __init__(self):
        super(TiDialog, self).__init__()
        loadUi('ti.ui', self)

        # precautions.ui에서 home 버튼에 클릭 이벤트를 연결
        self.timer.clicked.connect(self.start_timer_app)
        
    def start_timer_app(self):
        self.timer_app = TimerApp()
        # 새로운 레이아웃 형성
        layout = QVBoxLayout()
        layout.addWidget(self.timer_app)

        # 현재 위젯에 새로운 레이아웃 설정
        self.setLayout(layout)

        
        
class SettingDialog(QDialog):
    def __init__(self):
        super(SettingDialog, self).__init__()
        loadUi('set.ui', self)
        
        self.sets.clicked.connect(start)
        self.setn.clicked.connect(self.show_ti_dialog)
    """    
    def start_timer_app(self):
        self.timer_app = TimerApp()
        # 새로운 레이아웃 형성
        layout = QVBoxLayout()
        layout.addWidget(self.timer_app)

        # 현재 위젯에 새로운 레이아웃 설정
        self.setLayout(layout)
        
        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.timer_app)

        self.setLayout(layout)
        
    """    
    def show_ti_dialog(self):
        # precautions.ui를 불러와서 다이얼로그로 띄우기
        ti_dialog = TiDialog()
        ti_dialog.exec_()
        
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

def play_notification_sound():
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound('ok.wav')  # 알림음 파일명을 적절히 변경
    alert_sound.play()
    pygame.time.delay(1000)  # 1초 동안 대기

def measure_ear(duration, message, release_cam=True):
    global cap

    print(message)

    if duration == 3:
        print("3초 대기 후 알람이 울립니다.")
        time.sleep(3)
        play_notification_sound()
    elif duration == 5:
        print("5초 대기 후 알람이 울립니다.")
        time.sleep(5)
        play_notification_sound()

    print("캠을 켭니다.")
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    print("측정을 시작합니다.")

    ear_list = []

    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if faces:
            face = faces[0]
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]

            ear = calculate_EAR(left_eye + right_eye)
            ear_list.append(ear)

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        cv2.imshow("Frame", frame)
        cv2.waitKey(30)

    average_ear = sum(ear_list) / len(ear_list)
    print("Average EAR:", average_ear)
    print("측정이 완료되었습니다.")

    if release_cam:
        print("캠을 끕니다.")
        cap.release()
        cv2.destroyAllWindows()

    return average_ear

# 기타 설정
frames_to_measure = {"open": 3, "closed": 5}  # 수정: 눈을 뜬 상태와 감은 상태를 측정할 시간 (초)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def start():
    global EAR_THRESH
    open_ear = measure_ear(frames_to_measure["open"], "3초 후 측정이 시작됩니다.", release_cam=False)

    closed_ear = measure_ear(frames_to_measure["closed"], "5초 후 측정이 시작됩니다.")
    EAR_THRESH=closed_ear+((open_ear-closed_ear)/2)
    play_notification_sound()  # 마지막 알람 울리기

    print("눈을 뜨고 있는 상태의 EAR:", open_ear)
    print("눈을 감은 상태의 EAR:", closed_ear)
    
    print("캠을 끕니다.")
    cap.release()
    cv2.destroyAllWindows()
    
    return EAR_THRESH
     
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())