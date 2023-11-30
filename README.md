# Drowsiness detection
***
## Motive
: 공부를 하는 도중 졸리거나 잠이 드는 상황은 딱히 반길만한 상황은 아닙니다. 누군가 옆에 있다면 깨워줄 수 있겠지만 혼자서 공부하는 경우 그 상황을 기대할 수는 없습니다. 커피를 마시는 거나 세수를 하는 등 졸음을 이겨내기 위한 행동을 할 수는 있지만 졸음을 이겨내는 것은 쉽지 않습니다. 따라서 졸음을 감지해 알람을 울리는 시스템이 있다면 졸음을 깨는 데 도움이 될 것 같아 해당 프로그램을 제작하게 되었습니다. 

+ 기존에 있던 것에서 추가한 것은 **추가**로 표시하겠습니다.

  
## Description
: 얼굴 및 안구 검출을 하기 위해 Histogram of Oriented Gradients 기술과 학습된 Face landmark estimation 기법을 사용하였습니다. 조명 영향을 제거하기 위해선 원본 영상의 조명 채널을 분리해 역 조명을 쏘아 Grayscale 된 이미지와 합쳐주었고, 졸음 상태를 감지하기 위해선 Eye Aspect Ratio라는 개념을 사용하였습니다. 마지막으로 사용자의 졸음 위험 수준을 세 단계로 나눠 단계별로 차등 알람이 울리게 하였고, 단계를 나누는 과정에서 KNN 알고리즘을 사용했습니다. **공부를 하는데 있어 효율도 중요하기 때문에 타이머를 추가하여 일정 시간 동안 공부를 할 수 있도록 하였습니다.**


## Extracting face and eye region
+ 그레이스케일링 한 이미지에서 얼굴을 찾기 위해 HOG face pattern을 사용했습니다.
   
+ Face Landmark Estimation 알고리즘을 사용해 얼굴의 68개 랜드마크를 찾아냈습니다.
  
<img src="https://user-images.githubusercontent.com/36785390/52613175-3d6ade80-2ed0-11e9-9290-ee5dc2f2d525.png" width="30%">


## Preprocessing
+ 영상에 있어서 조명의 영향은 영상처리에 상당히 많은 영향을 끼칩니다. 특히 그라데이션 조명을 받았을 경우 에러를 일으키는 요소가 되기 때문에, 전처리 과정으로 영상에서 조명 영향을 받을 때 그 영향을 최소화하는 작업을 진행했습니다.
+ 전처리를 위해 영상에서 분리한 Lightness 채널을 반전시키고 Grayscale 된 원본 영상과 합성하여 Clear 한 Image를 만들었습니다.
    
     
## Drowsiness detection method
+ 이 프로젝트에서는 2016년 Tereza Soukupova & Jan ´ Cech에 의해 제시된 Eyes Aspect Ratio(이하 EAR) 방식을 사용합니다. EAR은 검출된 안구에 여섯 개의 (x, y) 좌표를 이용하여 계산됩니다.
  
<img src="https://user-images.githubusercontent.com/36785390/52702447-83eb3680-2fbf-11e9-985f-f96ec72f5b26.png" width="20%">
   
+ The EAR equation
   
<img src="https://user-images.githubusercontent.com/36785390/52702578-cb71c280-2fbf-11e9-9a06-d4434250d622.png" width ="30%">

+ Calculated EAR
<img src="https://user-images.githubusercontent.com/36785390/52702645-ee9c7200-2fbf-11e9-9757-975fa22da6e1.png" width="60%">

+ 계산된 EAR은 눈을 뜨고 있을 땐 0이 아닌 어떤 값을 갖게 되고, 눈을 감을 땐 0에 가까운 값을 갖습니다. 여기에 어떤 Constant로 Threshold를(졸음을 판단할 때 사용하는 임곗값) 설정할 시 그 값보다 EAR 값이 작아지는지 확인하는 방식으로 사용자가 졸고 있다고 판단할 수 있습니다.
+ 졸음 판별 시 양쪽 눈을 따로 검사할 필요는 없기 때문에 양쪽 눈 각각의 EAR 값을 평균 계산해서 사용하였습니다.
+ **Threshold 값은 프로그램 실행 시 눈을 떴을 때의 EAR 값(open_ear)과 눈을 감았을 때의 EAR 값(closed_ear)을 얻어 closed_ear+((open_ear-closed_ear)/2)라는 식을 통해 결정하였습니다.**


## Drowsiness level selection
+ 약 25프레임 동안 EAR 값이 Threshold보다 작으면 사용자가 졸고 있다고 판단하도록 설정하였습니다. 
+ 이 프로젝트에서는 졸음그래프를 기준으로 실제 졸음 단계를 결정하기 위해서 지도 학습(Supervised Learning) 알고리즘 중 하나인 K-Nearest Neighbor(이하 KNN) 알고리즘을 사용하였습니다.

+ Drowsiness levels are identified by the following conditions.
  1. The first alarm will sound(approximately 0.9 seconds) between level 1 and 2 of the drowsy phase.
  2. If you are dozing (sleeping and waking again and again) in less than 15 seconds, the drowsiness phase starts at level 1 and then the next alarm goes up to 0.
  3. The first alarm is level 2 and the second alarm is level 1 and the third alarm makes level 0 sound when driving drowsy between 15 and 30 seconds.
  4. If you have not been drowsy for more than 30 seconds, set level 2.
+ 졸음 단계는 눈을 감고 있는 시간과 졸음운전 전까지 눈을 뜨고 있던 시간에 따라 구분되고, 졸음 2 -> 0으로 갈수록 알람의 세기는 세집니다.
+ 설정한 Threshold 보다 작을 때 눈을 뜨고 있던 시간에 따라 구분되고, 졸음 2 -> 0으로 갈수록 알람의 세기는 세집니다.

  
. 1. Create arrays with random (x, y)-coordinates.
  
<img src="https://user-images.githubusercontent.com/36785390/52762829-82bc1700-305c-11e9-97cb-b41e35dfb9e6.png" width="30%">
  
  2. Labeling
<img src="https://user-images.githubusercontent.com/36785390/52762830-8485da80-305c-11e9-96db-f24a7a1ebdd6.png" width="40%">
  
  3. Define K value.
<img src="https://user-images.githubusercontent.com/36785390/52762904-e6dedb00-305c-11e9-952c-f201390eb9bd.png" width="50%">
  
  4. Test KNN algorithm.
<img src="https://user-images.githubusercontent.com/36785390/52762907-e8a89e80-305c-11e9-8928-9409bd4eaa7a.png" width="50%">
  
## Timer
+ **미시간 대학의 연구에 따르면, 공부를 집중할 수 있는 효과적인 시간은 25분이라고 합니다. 이를 볼 때 공부를 하는 데 있어 일정 시간을 정해두고 공부를 한 뒤 휴식을 취하고 다시 시작하는 것이 좋다고 생각되어 타이머를 추가하게 되었습니다.**
+ **원하는 시간을 선택하고 start를 누르면 정해둔 시간이 다 지난 뒤 알람을 울릴 것인지 말지 선택할 수 있습니다.**

## GUI
+ **사용자가 편하게 사용할 수 있도록 GUI를 추가하였습니다.**
---
+ **처음 시작**
<img width="50%" src="https://github.com/yu2842/sleep/assets/144086393/7b696aeb-d5b4-42d6-970a-ede08a7d4473"/>


---  
+ **precautions 버튼을 누른 경우**
+ **해당 프로그램을 사용할때의 주의 사항이 적혀 있습니다.**
+ **home 버튼을 통해 처음 시작으로 돌아갈 수 있습니다**
<img width="50%" src="https://github.com/yu2842/practice1/assets/144086393/267d24f9-c1f7-4125-bcf5-1e1d0a42b3ab"/>


---  
+ **'how to use' 버튼을 누른 경우**
+ **사용 할때 도움이 될 만한 것들이 적혀 있습니다.**
+ **home 버튼을 통해 처음 시작으로 돌아갈 수 있습니다.**
<img width="50%" src="https://github.com/yu2842/practice1/assets/144086393/6018acfd-3a01-4161-8301-54bb1ddc0c41"/>
  

---  
+ **처음 시작의 'start' 버튼을 누른 경우**
+ **start 버튼을 누를 경우 위에서 설명했던 Threshold 값을 얻는 작업을 수행합니다.**
+ **start 버튼을 누르기 전 next 버튼을 누르면 Threshold 값이 0으로 설정되므로 정상적인 졸음 판단을 할 수 없습니다.**
+ **Threshold 값을 결정 후 next 버튼을 누르면 timer로 이동합니다.**
<img width="50%" src="https://github.com/yu2842/practice1/assets/144086393/c2e681c5-a603-42d2-b484-9debd24d9cc6"/>
  

---  
+ **timer 버튼을 누르면 타이머가 뜹니다.**
<img width="50%" src="https://github.com/yu2842/practice1/assets/144086393/268681f3-e9ac-4301-8c25-35357ad3dfa6"/>


---  
+ **HH:MM:TT**  
+ **원하시는 시간을 작성 후 start 버튼을 누르면 졸음 판단 작업을 수행합니다.**
+ **시간이 다 지난 후 알람을 울릴지 말지 선택할 수 있습니다.**
+ **주의: pause 버튼이나 stop 버튼을 누른 후 캠을 누르고 'q' 키를 눌러 졸음 판단 작업을 종료하셔야합니다.**  
+ **졸음 판단 작업을 종료하지 않고 start 버튼을 다시 누르면 오류가 발생합니다.**
<img width="50%" src="https://github.com/yu2842/practice1/assets/144086393/7dc69de1-a57d-4764-a5ed-3ce1bd0678e4"/>
<img width="50%" src="https://github.com/yu2842/practice1/assets/144086393/60702dbb-f821-4209-a1f9-a070e1b55f00"/>
 

## File Description
1. **알람 소리**
   - alert_sound.wav
   - init_sound.mp3
   - nomal_alarm.wav
   - short_alarm.mp3
   - ok.wav
   - power_alarm.wav


2. **메인 코드(졸음 판단, 타이머, GUI 실행, EAR측정 등의 역할을 수행)**
   - eend.py
     

3. **QT designer로 만든 ui**
   - how.ui
   - main.ui
   - precautions.ui
   - set.ui
   - ti.ui


4. **영상에서 조명 영향을 받을 때 그 영향을 최소화하는 작업 수행**
   - light_remover.py


5. **졸음 단계를 결정하기 위해서 지도 학습**
   - make_train_data.py


6. **졸음 단계에 따른 알람 소리 선택**
   - ringing_alarm.py

7. **dlib에서 Face landmark estimation 기법을 사용하기 위한 자료**
   - shape_predictor_68_face_landmarks.dat





## Process
1. **처음 시작하는 경우 precautions 버튼과 how to use 버튼을 눌러 주의 사항과 사용법을 익힙니다.**
2. **사용 법을 알고 있거나 익혔을 경우 start 버튼을 누릅니다.**
3. **버튼을 누르면 Setting 창이 뜹니다. 다음과 같은 사항이 적혀 있습니다.**
   + **눈이 닫혔는지를 판단하기 위해 EAR 값을 측정해야 합니다.**
   + **준비가 되셨다면 start 버튼을 눌러주세요.**
   + **3초후 알람이 울립니다.**
   + **다시 알람이 울릴때까지 3초 동안 눈을 뜨고 있어주세요.**
   + **3초 후 알람이 울리면 5초 대기 후 다시 알람이 울릴때까지 5초 동안 눈을 감고 있어주세요.**
   + **다시 알람이 울리면 EAR 값 측정은 끝났습니다.**
   + **next 버튼을 눌러 drowsiness detection을 시작하세요.**
4. **해당 과정을 따르고 next 버튼을 누르면 timer로 이동합니다.**
5. **timer 버튼을 누르면 시간을 정할 수 있고 start 버튼을 누르면 졸음 감지 기능을 수행 할 수 있습니다.**
6. **start 버튼을 누르면 캠이 켜지는데, 이때 얼굴이 제대로 인식되고 있는지는 파란 네모와 초록 선, EAR 값을 보고 알 수 있습니다.**

   
   (**사진은 그림으로 대체합니다.**)

   
   <img width="50%" src="https://github.com/yu2842/practice1/assets/144086393/5d4a05f5-c78f-421d-8551-202569042ddb"/>

 
8. **pause 버튼의 경우 타이머가 일시 정지되며 이때 캠을 누른 후 q키를 눌러 졸음 감지 기능을 중지해야 합니다.**
9. **pause 버튼을 누르고 start 버튼을 누르면 정지되었던 타이머와 졸음 감지 기능이 다시 시작합니다.**
10. **stop 버튼의 경우 타이머가 정지되며 이때 캠을 누른 후 q키를 눌러 졸음 감지 기능을 중지해야 합니다.**
11. **프로그램을 완전히 종료하기 위해 모든 창을 꺼주세요.**


## LICENSE
**MIT**


**관련 정보는 LICENSE에서 확인**

## References
+ [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
+ [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
+ [Eye blink detection with OpenCV, Python, and dlib](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
+ [dlib install tutorial that I refer to](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)
+ [Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
+ [조명(Lighting)의 영향을 제거하는 방법](https://t9t9.com/60)
+ [Drowsiness driving detection system with OpenCV & KNN](https://github.com/woorimlee/drowsiness-detection)
+ [효율적인 공부방법, 어떻게 공부해야 할까?](http://www.hcnews.or.kr/news_gisa/gisa_view.htm?gisa_category=02010300&gisa_idx=10103)
