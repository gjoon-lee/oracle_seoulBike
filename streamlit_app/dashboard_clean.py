"""
서울시 따릉이 통합 분석 시스템
Seoul Bike Share Analysis System
Based on LightGBM 2-hour stockout prediction
"""

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
from datetime import datetime
import time
from api_client import BikeAPIClient

# 페이지 설정
st.set_page_config(
    page_title="서울시 따릉이 통합 분석 시스템",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS - Clean table style like reference
st.markdown("""
<style>
    .main { padding: 1rem; }
    
    /* Header style */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Metrics cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
    }
    
    .metric-value.red {
        color: #e53e3e;
    }
    
    /* Section headers */
    .section-header {
        background: #f7fafc;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 2rem 0 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    /* Status badges */
    .badge-empty {
        background: #e53e3e;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    .badge-warning {
        background: #ed8936;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    .badge-normal {
        background: #48bb78;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    /* Time indicators */
    .time-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .time-card.yellow {
        background: #fef5e7;
        border-color: #f6e05e;
    }
    
    .time-card.orange {
        background: #fed7d7;
        border-color: #fc8181;
    }
    
    .time-card.red {
        background: #fed7d7;
        border-color: #e53e3e;
    }
</style>
""", unsafe_allow_html=True)

# API 클라이언트
@st.cache_resource
def get_api_client():
    return BikeAPIClient()

api_client = get_api_client()

# 헤더
st.markdown("""
<div class='header-container'>
    <h1 style='text-align: center; margin: 0;'>🚴 서울시 따릉이 통합 분석 시스템</h1>
    <p style='text-align: center; margin-top: 0.5rem; opacity: 0.9;'>
        Seoul Public Bike Management Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# 탭 생성
tab1, tab2, tab3 = st.tabs(["실시간 현황", "AI 예측", "데이터 관리"])

with tab1:
    # 새로고침
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("🔄 새로고침"):
            st.cache_data.clear()
            st.rerun()
    with col1:
        st.markdown(f"**마지막 업데이트:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 데이터 가져오기
    try:
        stations_data = api_client.get_stations_status()
        predictions_data = api_client.get_predictions()
        
        if stations_data and predictions_data:
            stations_df = pd.DataFrame(stations_data['stations'])
            predictions_df = pd.DataFrame(predictions_data['predictions'])
            
            # 병합
            df = stations_df.merge(
                predictions_df[['station_id', 'stockout_probability', 'is_stockout_predicted']], 
                on='station_id', 
                how='left'
            )
            
            # 메트릭 계산
            total_stations = len(df)
            empty_now = len(df[df['available_bikes'] == 0])
            nearly_empty = len(df[df['available_bikes'] <= 2])
            avg_usage = (1 - df['available_bikes'] / df['station_capacity']).mean() * 100
            predicted_empty = len(df[df['is_stockout_predicted'] == 1])
            
            # 상단 메트릭 카드
            st.markdown("### 📊 현재 상태")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>전체 대여소</div>
                    <div class='metric-value'>{total_stations:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>비어있는 대여소</div>
                    <div class='metric-value red'>{empty_now}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>평균 사용률</div>
                    <div class='metric-value'>{avg_usage:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>경고 대여소</div>
                    <div class='metric-value red'>{predicted_empty}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # 빈 대여소 시간 분석 (2시간 후 예측 기반)
            st.markdown("""
            <div class='section-header'>
                빈 대여소 시간 분석
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            # 2시간 후 예측 기반 분류
            very_high_risk = len(df[df['stockout_probability'] >= 0.8])
            high_risk = len(df[(df['stockout_probability'] >= 0.6) & (df['stockout_probability'] < 0.8)])
            medium_risk = len(df[(df['stockout_probability'] >= 0.4) & (df['stockout_probability'] < 0.6)])
            low_risk = len(df[df['stockout_probability'] < 0.4])
            
            with col1:
                st.markdown(f"""
                <div class='time-card yellow'>
                    <h4>저위험</h4>
                    <h2>{low_risk}</h2>
                    <small>&lt;40% 확률</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='time-card orange'>
                    <h4>중위험</h4>
                    <h2>{medium_risk}</h2>
                    <small>40-60% 확률</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='time-card orange'>
                    <h4>고위험</h4>
                    <h2>{high_risk}</h2>
                    <small>60-80% 확률</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='time-card red'>
                    <h4>매우 위험</h4>
                    <h2>{very_high_risk}</h2>
                    <small>&gt;80% 확률</small>
                </div>
                """, unsafe_allow_html=True)
            
            # 대여소 실시간 현황 테이블
            st.markdown("""
            <div class='section-header'>
                대여소 실시간 현황
            </div>
            """, unsafe_allow_html=True)
            
            # 현재 비어있거나 거의 빈 대여소
            critical_df = df[(df['available_bikes'] <= 2)].copy()
            critical_df = critical_df.sort_values('available_bikes')
            
            if len(critical_df) > 0:
                # 표시할 컬럼 준비
                display_df = critical_df[['station_id', 'station_name', 'available_bikes', 
                                         'station_capacity', 'utilization_rate']].copy()
                
                # 구 정보 추가 (station_name에서 파싱 시도)
                display_df['구'] = '미포구'  # 기본값
                
                # 사용률 계산
                display_df['사용률'] = ((1 - display_df['available_bikes'] / display_df['station_capacity']) * 100).round(1).astype(str) + '%'
                
                # 상태 추가
                display_df['상태'] = display_df['available_bikes'].apply(
                    lambda x: '비어있음' if x == 0 else '거의없음' if x <= 2 else '정상'
                )
                
                # 컬럼명 변경
                display_df = display_df[['station_name', '구', 'available_bikes', 'station_capacity', '사용률', '상태']]
                display_df.columns = ['대여소명', '구', '자전거', '거치대', '사용률', '상태']
                
                # 테이블 표시
                st.dataframe(
                    display_df.head(20),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "상태": st.column_config.TextColumn(
                            "상태",
                            help="현재 자전거 상태"
                        ),
                    }
                )
            else:
                st.info("현재 비어있는 대여소가 없습니다.")
        
        else:
            st.error("데이터를 불러올 수 없습니다.")
    
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")

with tab2:
    st.markdown("### 🤖 AI 예측 경보 (2시간 후)")
    st.markdown("LightGBM 모델 기반 2시간 후 재고 부족 예측")
    
    try:
        high_risk_data = api_client.get_high_risk_stations(threshold=0.5)
        
        if high_risk_data and 'high_risk_stations' in high_risk_data:
            risk_stations = pd.DataFrame(high_risk_data['high_risk_stations'])
            
            if len(risk_stations) > 0:
                # 매우 위험 (>80%)
                st.markdown("#### 🔴 매우 위험 (80% 이상)")
                very_high = risk_stations[risk_stations['stockout_probability'] >= 0.8].copy()
                
                if len(very_high) > 0:
                    display = very_high[['station_id', 'current_available_bikes', 
                                        'station_capacity', 'stockout_probability']].copy()
                    display['stockout_probability'] = (display['stockout_probability'] * 100).round(1).astype(str) + '%'
                    display.columns = ['대여소 ID', '현재 자전거', '거치대', '2시간 후 재고부족 확률']
                    
                    st.dataframe(display.head(10), use_container_width=True, hide_index=True)
                else:
                    st.info("매우 위험한 대여소가 없습니다")
                
                # 위험 (60-80%)
                st.markdown("#### 🟠 위험 (60-80%)")
                high = risk_stations[(risk_stations['stockout_probability'] >= 0.6) & 
                                    (risk_stations['stockout_probability'] < 0.8)].copy()
                
                if len(high) > 0:
                    display = high[['station_id', 'current_available_bikes', 
                                  'station_capacity', 'stockout_probability']].copy()
                    display['stockout_probability'] = (display['stockout_probability'] * 100).round(1).astype(str) + '%'
                    display.columns = ['대여소 ID', '현재 자전거', '거치대', '2시간 후 재고부족 확률']
                    
                    st.dataframe(display.head(10), use_container_width=True, hide_index=True)
                else:
                    st.info("위험 수준 대여소가 없습니다")
            else:
                st.success("✅ 2시간 내 재고 부족이 예상되는 대여소가 없습니다")
        
    except Exception as e:
        st.error(f"예측 데이터 오류: {str(e)}")

with tab3:
    st.markdown("### 🔧 데이터 관리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 시스템 정보")
        st.info("""
        **LightGBM 모델**
        - 예측 대상: 2시간 후 재고 부족 (≤2대)
        - 정확도: 85.53%
        - ROC-AUC: 0.8955
        - 업데이트: 2025-08-19
        """)
    
    with col2:
        st.markdown("#### API 상태")
        if st.button("API 상태 확인"):
            try:
                health = api_client.get_system_health()
                if health:
                    st.success(f"✅ API 정상 작동: {health.get('status')}")
                else:
                    st.error("API 응답 없음")
            except:
                st.error("API 연결 실패")

# 자동 새로고침 (5분)
if st.sidebar.checkbox("자동 새로고침 (5분)", value=False):
    time.sleep(300)
    st.rerun()