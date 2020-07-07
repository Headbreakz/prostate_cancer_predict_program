```
└─prostate_cancer_predict_program
    │  image_change.py
    │  patient_DB.db
    │  program_ui.py
    │  README.md
    │  server_connect.py
    │  service.py
    │  test.txt
    │  tree_viex.txt
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
    ├─image
    │      768098d5b47a4b3e2f18993da3bda0a4.png
    │      768098d5b47a4b3e2f18993da3bda0a42020618.png
    │      78e15fff6ef70715b6f5b44caf905f97.png
    │      78fa6eadfc403f3440ef91db24d387b6.png
    │      79b3f2787261e5c96b3c9bf9a33c7537.png
    │      79b3f2787261e5c96b3c9bf9a33c753720206192020619.png
    │      79c8f162b231aebb6c639afa3f4395ae.png
    │      7a059d43b8f1c7c231a2257df8a1a9cb.png
    │      7b8d2f39e387bc0a54504377f8318247.png
    │      7c3ee27216049d5308310f02fba18dbb.png
    │      7c9f666901629ca31c8dad289aa18211.png
    │      7cbab4cb4f2b654dab01864409304a36.png
    │      7f46d3906c54e47d26aa7d3114b36a9d.png
    │      7f57053f651025603c6b4246acd0ba54.png
    │      7f8e10f3342e3763a4251fbde58ba92e.png
    │      7fdda19fabed1c92c7ec50194dc96371.png
    │      80175c7d26a0677a08f54c5b66af1555.png
    │      80175c7d26a0677a08f54c5b66af15552020618.png
    │      80983c7885cd02c9e9df539b58add687.png
    │      80983c7885cd02c9e9df539b58add6872020618.png
    │      80c676be159548673ff236d808209e33.png
    │      80f7cfb8ff20d9aa8eebcd541d0d5fd4.png
    │      80fb0f7ade621da69c54fad193c70212.png
    │      8143e5473a650fd79750a6b4cf7c52e1.png
    │      814e03e0dabe52a24961a88e331c43f7.png
    │      81b6b10c60f38572abf01c112ca59646.png
    │      81b6b10c60f38572abf01c112ca596462020618.png
    │      8245e742febd6314e31245fb410ac00c.png
    │      8299e52f9a53b39407d0950456562add.png
    │      82b8e31cf5bd59934af005c609d05e64.png
    │      83b8997aa1ebef2e386f3daf89226538.png
    │      84135442a04a1324138d1df78ed185d3.png
    │      8434a27f3a5aa65aac6fd2728c31297e.png
    │      8434a27f3a5aa65aac6fd2728c31297e2020620.png
    │      8488c41966af627575818380e131ffa4.png
    │      8488c41966af627575818380e131ffa42020618.png
    │      84a64d55903e7dfbad5e59693be25669.png
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







#### 전립선암 진단 보조 프로그램

---

1. 제작 배경 

   * 전립선 암 등급 예측 프로젝트 진행 중, 예측 모델을 사용하여 사용자가 결과를 편리하게 확인하기 위해 제작

   

2. 시스템 구조

   * PyQT를 사용하여 UI를 제작 , 환자 정보는 DB로 저장하여 보관, 전립선 생검 조직 사진은 저장소에 저장

   * GPU 기반으로 설계되어, AWS와 Socket 통신되도록 설계

   * AWS 내에 예측 모델 저장

     

3.  프로그램 기능

   ![program_ui](.\program_image\program_ui.png)

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