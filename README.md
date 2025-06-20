# ParkSmart Project
## 인하대학교 자동차 인공지능 프로젝트 : ParkSmart
### korean
실외 환경에서 주차장 여석 찾기 및 차종에 따른 주차장 안내 프로젝트

+ 프로젝트 목표
1. 주차장에 진입하는 차량의 차종, 브랜드를 판별한다.
2. 판단된 차종 및 브랜드를 통해 비어있는 주차공간으로 안내한다.

해당 프로젝트에서 1번 과제를 맡아 진행하였습니다.

## 차량 판별 AI 개요
+ 차종 5종을 분류한다. (트럭, 승용차, 버스, 이륜차, 전동 킥보드)
+ 브랜드 10종을 분류한다. (데이터셋에서 가장 많은 top 10개)
+ Python, Pytorch, torchvision
+ pretrained 된 ResNet18을 활용하여 분류 모델 구축

## 데이터 전처리
데이터셋은 [AI Hub의 자동차 차종/연식/번호판 인식용 영상](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=172)
[Roboflow kickboard](https://universe.roboflow.com/inha-univ-vgzgz/kickboard-ibhkj/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

+ 각 차종, 브랜드 간 데이터 수 차이로 인한 쏠림 현상 방지
+ 2000장을 넘어가는 sample은 random sampling / 2000장 이하의 sample은 data augmentation 진행
+ 
