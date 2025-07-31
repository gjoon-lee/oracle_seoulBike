# 실행 방법

1. 서울시 따릉이대여소 마스터 정보 API 키 발급 => KEY_BIKE_STATION_MASTER=(api키) 로 .env 파일에 저장
2. 서울시 공공자전거 실시간 대여정보 API 키 발급 => KEY_BIKE_LIST=(api키) 로 .env 파일에 저장
3. .venv 셋팅 => pip install -r requirements.txt
4. run api_test.py로 API 연결 확인하기
5. bikeList_load.py로 데이터 수집 (cntrl + c로 중단하기) => bike_fetch.log에서 데이터가 잘 수집되고 있는지 확인
6. eda.ipynb에서 데이터 분석 진행
