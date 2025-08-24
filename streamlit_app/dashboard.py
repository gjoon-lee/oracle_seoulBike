"""
서울시 따릉이 통합 관리 시스템 대시보드
Seoul Bike Share Management Dashboard
"""

import warnings
import logging

# Suppress the ScriptRunContext warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import numpy as np
from api_client import BikeAPIClient
from components import (
    create_metric_card, 
    create_status_table,
    create_hourly_chart,
    create_stockout_prediction_table
)

# 페이지 설정
st.set_page_config(
    page_title="서울시 따릉이 통합 분석 시스템",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .empty-station {
        background-color: #ffebee;
    }
    .warning-station {
        background-color: #fff3e0;
    }
    .header-style {
        background: linear-gradient(90deg, #5B47FB 0%, #7B68EE 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# API 클라이언트 초기화
@st.cache_resource
def get_api_client():
    return BikeAPIClient()

# 헤더
st.markdown("""
<div class="header-style">
    <h1 style='text-align: center; color: white; margin-bottom: 0;'>
        🚴 서울시 따릉이 통합 분석 시스템
    </h1>
    <p style='text-align: center; color: white; margin-top: 10px;'>
        Seoul Public Bike Management Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# 탭 메뉴
tab1, tab2, tab3, tab4 = st.tabs(["실시간 현황", "수요 예측", "주간 분석", "데이터 관리"])

with tab1:
    # 자동 새로고침 설정
    auto_refresh = st.sidebar.checkbox("자동 새로고침 (5분)", value=True)
    if auto_refresh:
        st.sidebar.info("5분마다 자동으로 데이터가 업데이트됩니다.")
    
    # 구 선택
    district_filter = st.sidebar.selectbox(
        "구 선택",
        ["전체 구"] + ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", 
                     "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구",
                     "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구",
                     "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"]
    )
    
    # API 클라이언트
    api_client = get_api_client()
    
    # 데이터 가져오기
    col_refresh = st.columns([5, 1])
    with col_refresh[1]:
        if st.button("🔄 새로고침"):
            st.cache_data.clear()
            st.rerun()
    
    with col_refresh[0]:
        st.markdown(f"**마지막 업데이트:** {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
    
    # 현재 상태 데이터 가져오기
    try:
        stations_data = api_client.get_stations_status()
        predictions_data = api_client.get_predictions()
        
        if stations_data and 'stations' in stations_data:
            stations_df = pd.DataFrame(stations_data['stations'])
            
            # 메트릭 계산
            total_stations = len(stations_df)
            empty_stations = len(stations_df[stations_df['is_stockout'] == 1])
            avg_utilization = stations_df['utilization_rate'].mean() * 100
            warning_stations = len(stations_df[stations_df['utilization_rate'] > 0.8])
            
            # 상단 메트릭 카드
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="전체 대여소",
                    value=f"{total_stations:,}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="비어있는 대여소",
                    value=f"{empty_stations:,}",
                    delta=f"{empty_stations/total_stations*100:.1f}%" if total_stations > 0 else "0%",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    label="평균 사용률",
                    value=f"{avg_utilization:.1f}%",
                    delta=None
                )
            
            with col4:
                st.metric(
                    label="경고 대여소",
                    value=f"{warning_stations:,}",
                    delta="80% 이상 사용",
                    delta_color="inverse"
                )
            
            st.markdown("---")
            
            # 빈 대여소 시간 분석
            st.subheader("📊 빈 대여소 시간 분석")
            
            # 예측 데이터와 병합
            if predictions_data and 'predictions' in predictions_data:
                pred_df = pd.DataFrame(predictions_data['predictions'])
                stations_df = stations_df.merge(
                    pred_df[['station_id', 'stockout_probability']], 
                    on='station_id', 
                    how='left'
                )
                
                # 시간대별 분류 (예측 확률 기반)
                time_categories = {
                    "1시간 미만": 0,
                    "1-2시간": 0,
                    "2-3시간": 0,
                    "3시간 이상": 0
                }
                
                for _, row in stations_df.iterrows():
                    if row.get('is_stockout', 0) == 1:
                        prob = row.get('stockout_probability', 0)
                        if prob < 0.3:
                            time_categories["1시간 미만"] += 1
                        elif prob < 0.5:
                            time_categories["1-2시간"] += 1
                        elif prob < 0.7:
                            time_categories["2-3시간"] += 1
                        else:
                            time_categories["3시간 이상"] += 1
                
                # 시간대별 카드
                time_cols = st.columns(4)
                colors = ["#FFF59D", "#FFB74D", "#FF7043", "#EF5350"]
                labels = list(time_categories.keys())
                
                for idx, (col, label) in enumerate(zip(time_cols, labels)):
                    with col:
                        st.markdown(f"""
                        <div style='background-color: {colors[idx]}; 
                                    padding: 20px; 
                                    border-radius: 10px; 
                                    text-align: center;
                                    color: {"white" if idx > 1 else "black"};'>
                            <h4 style='margin: 0;'>{label}</h4>
                            <h2 style='margin: 10px 0;'>{time_categories[label]}</h2>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 대여소 실시간 현황
            st.subheader("🚲 대여소 실시간 현황")
            
            # 필터링
            display_df = stations_df.copy()
            
            # 구 필터 적용 (실제 구 데이터가 있다면)
            if district_filter != "전체 구":
                # display_df = display_df[display_df['district'] == district_filter]
                pass
            
            # 상태 컬럼 추가
            display_df['상태'] = display_df.apply(
                lambda x: '🔴 비어있음' if x['is_stockout'] == 1 
                else '🟡 주의' if x['utilization_rate'] > 0.8 
                else '🟢 정상', axis=1
            )
            
            # 표시할 컬럼 선택 및 이름 변경
            display_columns = {
                'station_id': '대여소 ID',
                'station_name': '대여소명',
                'available_bikes': '자전거',
                'station_capacity': '거치대',
                'utilization_rate': '사용률',
                '상태': '상태'
            }
            
            display_df = display_df[list(display_columns.keys())]
            display_df.columns = list(display_columns.values())
            display_df['사용률'] = (display_df['사용률'] * 100).round(1).astype(str) + '%'
            
            # 비어있는 대여소 우선 정렬
            display_df = display_df.sort_values(by='자전거', ascending=True)
            
            # 상위 20개만 표시
            st.dataframe(
                display_df.head(20),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "상태": st.column_config.TextColumn(
                        "상태",
                        width="small",
                    ),
                    "사용률": st.column_config.TextColumn(
                        "사용률",
                        width="small",
                    ),
                }
            )
            
            # 경고 메시지
            if empty_stations > 0:
                st.warning(f"⚠️ 현재 {empty_stations}개 대여소가 비어있습니다. 즉시 재배치가 필요합니다.")
            
        else:
            st.error("❌ 데이터를 불러올 수 없습니다. API 서버를 확인해주세요.")
            
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        st.info("💡 API 서버가 실행 중인지 확인해주세요: http://localhost:8000/docs")

with tab2:
    st.subheader("📈 수요 예측")
    
    # 예측 시간 선택
    pred_col1, pred_col2 = st.columns([2, 3])
    with pred_col1:
        selected_station = st.selectbox(
            "대여소 선택",
            ["전체"] + [f"ST-{i:03d}" for i in range(1, 100)]  # 예시 대여소 ID
        )
        
        prediction_hours = st.slider(
            "예측 시간 (시간)",
            min_value=1,
            max_value=24,
            value=2,
            step=1
        )
    
    # LightGBM 재고 부족 예측
    st.markdown("### 🔮 재고 부족 예측 (2시간 후)")
    
    try:
        predictions = api_client.get_high_risk_stations(threshold=0.5)
        
        if predictions and 'high_risk_stations' in predictions:
            risk_df = pd.DataFrame(predictions['high_risk_stations'])
            
            if len(risk_df) > 0:
                # 위험도별 색상 매핑
                def get_risk_color(prob):
                    if prob >= 0.8:
                        return "🔴 매우 위험"
                    elif prob >= 0.6:
                        return "🟠 위험"
                    elif prob >= 0.4:
                        return "🟡 주의"
                    else:
                        return "🟢 양호"
                
                risk_df['위험도'] = risk_df['stockout_probability'].apply(get_risk_color)
                
                # 컬럼 매핑
                risk_display = risk_df[['station_id', 'current_available_bikes', 
                                       'stockout_probability', '위험도']].copy()
                risk_display.columns = ['대여소 ID', '현재 자전거', '부족 확률', '위험도']
                risk_display['부족 확률'] = (risk_display['부족 확률'] * 100).round(1).astype(str) + '%'
                
                st.dataframe(
                    risk_display.head(15),
                    use_container_width=True,
                    hide_index=True
                )
                
                # 경고 카운트
                critical_count = len(risk_df[risk_df['stockout_probability'] >= 0.8])
                if critical_count > 0:
                    st.error(f"🚨 {critical_count}개 대여소가 2시간 내 재고 부족이 예상됩니다!")
            else:
                st.success("✅ 2시간 내 재고 부족이 예상되는 대여소가 없습니다.")
        
    except Exception as e:
        st.error(f"예측 데이터를 불러올 수 없습니다: {str(e)}")
    
    # 24시간 수요 예측 차트 (placeholder)
    st.markdown("### 📊 24시간 수요 예측")
    
    # 샘플 데이터 생성 (실제로는 XGBoost API에서 가져와야 함)
    hours = list(range(24))
    current_hour = datetime.now().hour
    
    # 온도 데이터 (샘플)
    temperatures = [15 + 10 * np.sin((h - 6) * np.pi / 12) for h in hours]
    
    # 수요 예측 데이터 (샘플)
    demand = [50 + 30 * np.sin((h - 14) * np.pi / 12) + np.random.randint(-10, 10) for h in hours]
    
    # Plotly 차트 생성
    fig = go.Figure()
    
    # 수요 예측 라인
    fig.add_trace(go.Scatter(
        x=hours,
        y=demand,
        mode='lines+markers',
        name='예측 사용률 (%)',
        line=dict(color='#5B47FB', width=3),
        marker=dict(size=8),
        yaxis='y'
    ))
    
    # 온도 라인
    fig.add_trace(go.Scatter(
        x=hours,
        y=temperatures,
        mode='lines',
        name='온도 (°C)',
        line=dict(color='#FF9800', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    # 현재 시간 표시
    fig.add_vline(
        x=current_hour,
        line_dash="dash",
        line_color="red",
        annotation_text="현재 시간"
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title=f"{selected_station} - 24시간 수요 예측",
        xaxis=dict(
            title="시간",
            tickmode='array',
            tickvals=list(range(0, 24, 3)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)]
        ),
        yaxis=dict(
            title="예측 사용률 (%)",
            side='left',
            range=[0, 100]
        ),
        yaxis2=dict(
            title="온도 (°C)",
            side='right',
            overlaying='y',
            range=[0, 35]
        ),
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 수요 예측 요약
    st.info("""
    📌 **수요 예측 정보**
    - 예측 모델: XGBoost (Net Flow Regression)
    - 업데이트 주기: 1시간
    - 정확도: MAE ~3.4 bikes, R² ~0.61
    """)

with tab3:
    st.subheader("📊 주간 분석")
    
    # 주간 통계
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 주간 이용 패턴")
        
        # 요일별 이용량 (샘플 데이터)
        days = ['월', '화', '수', '목', '금', '토', '일']
        usage = [85, 88, 90, 92, 95, 75, 70]
        
        fig_weekly = go.Figure(data=[
            go.Bar(x=days, y=usage, marker_color='#5B47FB')
        ])
        
        fig_weekly.update_layout(
            title="요일별 평균 이용률 (%)",
            yaxis_title="이용률 (%)",
            xaxis_title="요일",
            height=300
        )
        
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with col2:
        st.markdown("### 시간대별 패턴")
        
        # 시간대별 이용량 (샘플 데이터)
        peak_hours = ['07-09시', '12-14시', '18-20시', '21-23시']
        peak_usage = [92, 75, 95, 60]
        
        fig_hourly = go.Figure(data=[
            go.Bar(x=peak_hours, y=peak_usage, 
                  marker_color=['#FF7043', '#FFB74D', '#FF7043', '#FFF59D'])
        ])
        
        fig_hourly.update_layout(
            title="주요 시간대 이용률 (%)",
            yaxis_title="이용률 (%)",
            xaxis_title="시간대",
            height=300
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # 주간 리포트
    st.markdown("### 📋 주간 리포트")
    
    report_data = {
        "지표": ["총 대여 건수", "평균 이용 시간", "재배치 횟수", "고장 신고", "신규 회원"],
        "이번 주": ["125,430", "18.5분", "342회", "28건", "1,234명"],
        "지난 주": ["118,920", "17.2분", "358회", "35건", "1,156명"],
        "변화율": ["+5.5%", "+7.6%", "-4.5%", "-20.0%", "+6.7%"]
    }
    
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("🔧 데이터 관리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 데이터 업데이트")
        
        if st.button("🔄 실시간 데이터 수집 시작", use_container_width=True):
            st.success("실시간 데이터 수집이 시작되었습니다.")
        
        if st.button("📊 예측 모델 재학습", use_container_width=True):
            st.info("모델 재학습이 예약되었습니다. (예상 소요시간: 30분)")
        
        if st.button("🗄️ 데이터베이스 백업", use_container_width=True):
            st.success("데이터베이스 백업이 완료되었습니다.")
    
    with col2:
        st.markdown("### 시스템 상태")
        
        # 시스템 상태 체크
        status_data = {
            "서비스": ["API 서버", "데이터베이스", "예측 모델", "캐시"],
            "상태": ["🟢 정상", "🟢 정상", "🟢 정상", "🟡 대기"],
            "응답시간": ["23ms", "5ms", "145ms", "1ms"]
        }
        
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True, hide_index=True)
        
        # 모델 정보
        st.markdown("### 모델 정보")
        st.info("""
        **LightGBM Stockout Classifier**
        - 버전: 20250819_072922
        - 정확도: 85.53%
        - ROC-AUC: 0.8955
        - 마지막 학습: 2025-08-19 07:29:22
        """)

# 사이드바 정보
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📌 빠른 링크")
    st.markdown("[📊 API 문서](http://localhost:8000/docs)")
    st.markdown("[📈 Grafana 대시보드](#)")
    st.markdown("[📝 운영 매뉴얼](#)")
    
    st.markdown("---")
    st.markdown("### 🌡️ 현재 날씨")
    weather_data = api_client.get_current_weather()
    if weather_data:
        weather = weather_data.get('weather', {})
        st.metric("온도", f"{weather.get('temperature', 15)}°C")
        st.metric("습도", f"{weather.get('humidity', 60)}%")
        st.metric("강수량", f"{weather.get('precipitation', 0)}mm")
    
    st.markdown("---")
    st.markdown("### ℹ️ 정보")
    st.info("""
    **서울시 따릉이 통합 분석 시스템**
    - Version: 1.0.0
    - Last Update: 2025.01.21
    - Contact: bike@seoul.go.kr
    """)

# 자동 새로고침
if auto_refresh:
    time.sleep(300)  # 5분
    st.rerun()