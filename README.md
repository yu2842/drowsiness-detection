# Drowsiness detection
***
## Motive
: 공부를 하는 도중 졸리거나 잠이 드는 상황은 딱히 반길만한 상황은 아닙니다. 누군가 옆에 있다면 깨워줄 수 있겠지만 혼자서 공부하는 경우 그 상황을 기대할 수는 없습니다. 커피를 마시는 거나 세수를 하는 등 졸음을 이겨내기 위한 행동을 할 수는 있지만 졸음을 이겨내는 것은 쉽지 않습니다. 따라서 졸음을 감지해 알람을 울리는 시스템이 있다면 졸음을 깨는 데 도움이 될 것 같아 해당 프로그램을 제작하게 되었습니다. 
  
## Description
: 얼굴 및 안구 검출을 하기 위해 **Histogram of Oriented Gradients 기술과 학습된 Face landmark estimation 기법**을 사용하였습니다. **조명 영향을 제거하기 위해선 원본 영상의 조명 채널을 분리해 역 조명을 쏘아 Grayscale 된 이미지와 합쳐**주었고, 졸음 상태를 감지하기 위해선 **Eye Aspect Ratio**라는 개념을 사용하였습니다. 마지막으로 사용자의 **졸음 위험 수준을 세 단계로 나눠** 단계별로 차등 알람이 울리게 하였고, 단계를 나누는 과정에서 KNN 알고리즘을 사용했습니다. 
    
## System diagram
 




## Extracting face and eye region
+ 그레이스케일링 한 이미지에서 얼굴을 찾기 위해 **HOG face pattern**을 사용했습니다.
   
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

+ **계산된 EAR은 눈을 뜨고 있을 땐 0이 아닌 어떤 값을 갖게 되고, 눈을 감을 땐 0에 가까운 값**을 갖습니다. 여기에 어떤 Constant로 **Threshold**를(졸음운전을 판단할 때 사용하는 임곗값) 설정할 시 그 값보다 EAR 값이 작아지는지 확인하는 방식으로 사용자가 졸고 있다고 판단할 수 있습니다.
+ 졸음 판별 시 양쪽 눈을 따로 검사할 필요는 없기 때문에 양쪽 눈 각각의 EAR 값을 평균 계산해서 사용하였습니다.
+ **Threshold** 값은 프로그램 실행 시 눈을 떴을 때의 EAR 값(open_ear)과 눈을 감았을 때의 EAR 값(closed_ear)을 얻어 closed_ear+((open_ear-closed_ear)/2)라는 식을 통해 결정하였습니다.
+ 설정한 Threshold 보다 작을 때는(눈 크기가 작아졌을 때) 사용자가 졸린 상태인 것으로 판단, 사용자가 졸려 하는지에 관심을 뒀기 때문에 완전 수면에 빠지지 않더라도 알람이 울립니다.

  
## Drowsiness level selection
+ 약 25프레임 동안 EAR 값이 Threshold보다 작으면 운전자가 졸음운전 중이라고 판단하도록 설정하였습니다. 
+ 이 프로젝트에서는 졸그래프를 기준으로 실제 졸음 단계를 결정하기 위해서 지도 학습(Supervised Learning) 알고리즘 중 하나인 K-Nearest Neighbor(이하 KNN) 알고리즘을 사용하였습니다.
  
. 1. Create arrays with random (x, y)-coordinates.
  
<img src="https://user-images.githubusercontent.com/36785390/52762829-82bc1700-305c-11e9-97cb-b41e35dfb9e6.png" width="30%">
  
  2. Labeling
<img src="https://user-images.githubusercontent.com/36785390/52762830-8485da80-305c-11e9-96db-f24a7a1ebdd6.png" width="40%">
  
  3. Define K value.
<img src="https://user-images.githubusercontent.com/36785390/52762904-e6dedb00-305c-11e9-952c-f201390eb9bd.png" width="50%">
  
  4. Test KNN algorithm.
<img src="https://user-images.githubusercontent.com/36785390/52762907-e8a89e80-305c-11e9-8928-9409bd4eaa7a.png" width="50%">
  
  
## Synthesis
<img src="https://user-images.githubusercontent.com/36785390/52762972-36bda200-305d-11e9-99a6-314dfae8f3c7.png" width="50%">

## Test
+ Before applying preprocessing
+ 전처리 전 시연 영상
[![BeforePreprocessing](https://img.youtube.com/vi/8yLHAP6gmOA/0.jpg)](https://www.youtube.com/watch?v=8yLHAP6gmOA)
+ After applying preprocessing
+ 전처리 후 시연 영상
[![AfterPreprocessing](https://img.youtube.com/vi/7iCVzF3LI6o/0.jpg)](https://www.youtube.com/watch?v=7iCVzF3LI6o)

  
## Execution
+ I run drowsiness_detector.ipynb just typing CTRL+ENTER.
+ 전 jupyter notebook을 사용했기 때문에 일단 업로드 해두었습니다. 파이썬으로 실행하셔도 됩니다.
  
## References
+ [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
+ [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
+ [Eye blink detection with OpenCV, Python, and dlib](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
+ [dlib install tutorial that I refer to](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)
+ [Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
+ [조명(Lighting)의 영향을 제거하는 방법](https://t9t9.com/60)
