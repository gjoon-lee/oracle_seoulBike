# 실행 방법

## 초기 설정

1. 서울시 따릉이대여소 마스터 정보 API 키 발급 => KEY_BIKE_STATION_MASTER=(api키) 로 .env 파일에 저장
2. 서울시 공공자전거 실시간 대여정보 API 키 발급 => KEY_BIKE_LIST=(api키) 로 .env 파일에 저장
3. .venv 셋팅 => pip install -r requirements.txt
4. run api_test.py로 API 연결 확인하기
5. bikeList_load.py로 데이터 수집 (cntrl + c로 중단하기) => bike_fetch.log에서 데이터가 잘 수집되고 있는지 확인
6. eda.ipynb에서 데이터 분석 진행

## 실시간 예측 API 실행

```bash
# 1. FastAPI 서버 실행
cd realtime_prediction
python main.py
# API 문서: http://localhost:8000/docs
```

## Streamlit 대시보드 실행

```bash
# 1. 새 터미널에서 Streamlit 실행
.venv\Scripts\streamlit.exe run streamlit_app\dashboard.py

# 또는 Linux/Mac:
source .venv/bin/activate
streamlit run streamlit_app/dashboard.py
```

대시보드는 http://localhost:8501 에서 실행됩니다.

### 대시보드 주요 기능
- **실시간 현황**: 전체 대여소 상태 모니터링, 빈 대여소 시간 분석
- **수요 예측**: LightGBM 2시간 재고부족 예측, XGBoost 24시간 수요 예측
- **주간 분석**: 요일별/시간대별 이용 패턴 분석
- **데이터 관리**: 실시간 데이터 수집, 모델 재학습, 시스템 상태 확인
