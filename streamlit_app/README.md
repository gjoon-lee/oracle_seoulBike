# 서울시 따릉이 통합 분석 시스템 대시보드

Seoul Public Bike (따릉이) Management Dashboard - Real-time monitoring and predictive analytics system.

## 주요 기능

### 1. 실시간 현황 모니터링
- 전체 대여소 상태 실시간 추적
- 비어있는 대여소 즉시 확인
- 구별 필터링 및 검색 기능
- 재고 부족 시간 예측 (1시간 미만, 1-2시간, 2-3시간, 3시간 이상)
- 색상 코드 경고 시스템 (정상/주의/위험)

### 2. 수요 예측 (AI 기반)
- **LightGBM 재고 부족 예측**: 2시간 후 stockout 확률 계산
- **XGBoost 수요 예측**: 24시간 net flow 예측
- 온도 등 날씨 변수 연동
- 시간대별 패턴 분석

### 3. 주간 분석
- 요일별 이용 패턴
- 시간대별 피크 분석
- 주간 KPI 리포트
- 전주 대비 성장률

### 4. 데이터 관리
- 실시간 데이터 수집 제어
- 모델 재학습 스케줄링
- 시스템 상태 모니터링
- 데이터베이스 백업

## 설치 및 실행

### 요구사항
- Python 3.9+
- PostgreSQL database
- FastAPI backend running on port 8000

### 설치
```bash
# 1. 필요 패키지 설치
pip install -r requirements.txt

# 2. API 서버 실행 (별도 터미널)
cd ../realtime_prediction
python main.py

# 3. Streamlit 대시보드 실행
streamlit run dashboard.py
```

### 빠른 시작 (Shell Script)
```bash
# Linux/Mac
chmod +x run_dashboard.sh
./run_dashboard.sh

# Windows
bash run_dashboard.sh
```

## 대시보드 구성

### 메인 화면 구조
```
┌─────────────────────────────────────────┐
│      🚴 서울시 따릉이 통합 분석 시스템      │
├─────────────────────────────────────────┤
│ [실시간 현황] [수요 예측] [주간 분석] [관리] │
├─────────────────────────────────────────┤
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│  │전체   │ │비어  │ │평균  │ │경고  │  │
│  │2,802 │ │ 272  │ │97.9% │ │  15  │  │
│  └──────┘ └──────┘ └──────┘ └──────┘  │
├─────────────────────────────────────────┤
│         빈 대여소 시간 분석               │
│  [1시간] [1-2시간] [2-3시간] [3시간+]    │
├─────────────────────────────────────────┤
│         대여소 실시간 현황 테이블          │
│  ID | 대여소명 | 자전거 | 사용률 | 상태  │
└─────────────────────────────────────────┘
```

## API 연동

대시보드는 FastAPI 백엔드와 다음 엔드포인트를 통해 통신합니다:

- `GET /api/stations/status` - 전체 대여소 상태
- `GET /api/predictions/stockout` - 재고 부족 예측
- `GET /api/predictions/high-risk` - 고위험 대여소 목록
- `GET /api/weather/current` - 현재 날씨
- `GET /api/statistics` - 시스템 통계

## 커스터마이징

### 테마 설정
`.streamlit/config.toml` 파일 생성:
```toml
[theme]
primaryColor = "#5B47FB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#262730"
font = "sans serif"
```

### 새로고침 주기 변경
```python
# dashboard.py line 556
time.sleep(300)  # 5분 → 원하는 초 단위로 변경
```

## 문제 해결

### API 연결 오류
```
❌ 데이터를 불러올 수 없습니다. API 서버를 확인해주세요.
```
해결: FastAPI 서버가 http://localhost:8000 에서 실행 중인지 확인

### 데이터 없음
```
✅ 2시간 내 재고 부족이 예상되는 대여소가 없습니다.
```
정상: 현재 모든 대여소가 안정적인 상태

## 스크린샷

### 실시간 현황
- 전체 대여소 메트릭 카드
- 시간별 빈 대여소 분류
- 실시간 상태 테이블

### 수요 예측
- LightGBM 2시간 stockout 예측
- 24시간 수요 차트
- 위험도별 대여소 목록

## 기술 스택
- **Frontend**: Streamlit 1.31.0
- **Visualization**: Plotly 5.18.0
- **Data Processing**: Pandas 2.0.3
- **API Client**: Requests 2.31.0
- **Backend**: FastAPI (별도 실행)
- **ML Models**: LightGBM, XGBoost

## 라이선스
Seoul Metropolitan Government

## 연락처
- 시스템 문의: bike@seoul.go.kr
- 기술 지원: 02-120