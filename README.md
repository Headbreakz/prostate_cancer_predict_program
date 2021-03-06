```
└─prostate_cancer_predict_program
    │  image_change.py
    │  patient_DB.db
    │  program_ui.py
    │  README.md
    │  server_connect.py
    │  service.py
    │  
    ├─.idea
    │  │  .gitignore
    │  │  misc.xml
    │  │  modules.xml
    │  │  QT.iml
    │  │  vcs.xml
    │  │  workspace.xml
    │  │  
    │  └─inspectionProfiles
    │          profiles_settings.xml
    │          
    ├─.ipynb_checkpoints
    ├─.vscode
    │      settings.json        
    │      
    ├─program_image
    │      program_ui.png
    │      
    └─__pycache__
            image_change.cpython-37.pyc
            predict2.cpython-37.pyc
            server_connect.cpython-37.pyc
            service.cpython-37.pyc
            ui.cpython-37.pyc
```

service.py 실행 



#### 전립선암 진단 보조 프로그램

---

1. 제작 배경 

   * 전립선 암 등급 예측 프로젝트 진행 중, 예측 모델을 사용하여 사용자가 결과를 편리하게 확인하기 위해 제작

   

2. 시스템 구조

   * PyQT를 사용하여 UI를 제작 , 환자 정보는 DB로 저장하여 보관, 전립선 생검 조직 사진은 저장소에 저장

   * GPU 기반으로 설계되어, AWS와 Socket 통신되도록 설계

   * AWS 내에 예측 모델 저장

     

3.  프로그램 기능

   ![program_ui](./program_image/program_ui.png)

   * 환자 검색 기능

     * 날짜, 검색 기능을 통해 환자를 검색 할 수 있다.

       

   * 전립선 생검 조직 보기

     * 선택한 환자의 전립선 생점 조직을 확인 할 수 있다.

       

   * 전립선 암 등급 예측

     * Prediction 기능을 통해 전립선 암 등급을 예측 할 수있다.

       

   * 전립선 암 부위 예측

     * Search 기능을 통해 가장 문제가 되는 Gleason score 4,5 부분을 찾을 수 있다.

       

   * 저장 기능

     * 현재 화면에 보여지는 사진을 저장 할 수 있다.