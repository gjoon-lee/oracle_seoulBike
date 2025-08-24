"""
서울시 따릉이 통합 분석 시스템
Seoul Bike Share Analysis System
"""

import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
from api_client import BikeAPIClient

# 페이지 설정
st.set_page_config(
    page_title="서울시 따릉이 통합 분석 시스템",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일 - Force light theme and proper contrast
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: #ffffff;
    }
    
    .main { 
        padding: 0.5rem; 
        background-color: #f8f9fa;
    }
    
    /* Force table text to be dark */
    .stDataFrame div[data-testid="stDataFrameResizable"] * {
        color: #000000 !important;
    }
    
    /* Table headers */
    .stDataFrame th {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
        font-weight: 600 !important;
    }
    
    /* Table cells */
    .stDataFrame td {
        color: #262730 !important;
        background-color: white !important;
    }
    
    /* Metrics text fix */
    [data-testid="metric-container"] {
        background-color: white !important;
    }
    
    [data-testid="metric-container"] label {
        color: #262730 !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #262730 !important;
    }
    
    /* 헤더 스타일 */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* 메트릭 카드 */
    .metrics-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        flex: 1;
        background: white;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .metric-label {
        color: #718096;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2d3748;
    }
    
    .metric-value.danger {
        color: #e53e3e;
    }
    
    .metric-value.warning {
        color: #ed8936;
    }
    
    /* 섹션 헤더 */
    .section-header {
        background: white;
        padding: 0.75rem 1rem;
        border-left: 4px solid #667eea;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* 위험도 카드 */
    .risk-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .risk-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 2px solid;
    }
    
    .risk-card.low {
        border-color: #48bb78;
        background: #f0fff4;
    }
    
    .risk-card.medium {
        border-color: #ed8936;
        background: #fffaf0;
    }
    
    .risk-card.high {
        border-color: #e53e3e;
        background: #fff5f5;
    }
    
    .risk-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .risk-card p {
        margin: 0.5rem 0 0 0;
        color: #718096;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# API 클라이언트 초기화
@st.cache_resource
def get_api_client():
    return BikeAPIClient()

api_client = get_api_client()

# 헤더
st.markdown("""
<div class='header-container'>
    <h1 style='margin: 0; font-size: 2rem;'>🚴 서울시 따릉이 통합 분석 시스템</h1>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>실시간 현황 및 AI 예측 대시보드</p>
</div>
""", unsafe_allow_html=True)

# 지역 필터 및 새로고침
col1, col2, col3 = st.columns([2, 8, 2])
with col1:
    districts = ['전체', '강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', 
                 '금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', 
                 '서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구', 
                 '은평구', '종로구', '중구', '중랑구']
    selected_district = st.selectbox("지역 선택", districts, key="district_filter")

with col3:
    if st.button("🔄 새로고침", type="primary"):
        st.cache_data.clear()
        st.rerun()

# 데이터 가져오기
@st.cache_data(ttl=60)  # 1분 캐시
def fetch_all_data():
    stations = api_client.get_stations_status()
    predictions = api_client.get_predictions()
    high_risk = api_client.get_high_risk_stations(threshold=0.5)
    return stations, predictions, high_risk

try:
    stations_data, predictions_data, high_risk_data = fetch_all_data()
    
    if stations_data and predictions_data:
        stations_df = pd.DataFrame(stations_data['stations'])
        predictions_df = pd.DataFrame(predictions_data['predictions'])
        
        # 데이터 병합
        df = stations_df.merge(
            predictions_df[['station_id', 'stockout_probability', 'is_stockout_predicted']], 
            on='station_id', 
            how='left'
        )
        
        # 지역 필터 적용
        if selected_district != '전체':
            df = df[df['station_name'].str.contains(selected_district, na=False)]
        
        # 주요 지표 계산
        total_stations = len(df)
        empty_stations = len(df[df['available_bikes'] == 0])
        warning_stations = len(df[df['available_bikes'] <= 2])
        avg_usage = ((1 - df['available_bikes'] / df['station_capacity']) * 100).mean()
        predicted_stockout = len(df[df['is_stockout_predicted'] == 1])
        
        # 메트릭 표시
        st.markdown(f"""
        <div class='metrics-row'>
            <div class='metric-card'>
                <div class='metric-label'>전체 대여소</div>
                <div class='metric-value'>{total_stations:,}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>비어있는 대여소</div>
                <div class='metric-value danger'>{empty_stations}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>평균 사용률</div>
                <div class='metric-value'>{avg_usage:.1f}%</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>2시간 후 예측</div>
                <div class='metric-value warning'>{predicted_stockout}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 빈 대여소 시간 분석 (현재 비어있는 대여소들의 지속 시간)
        st.markdown("<div class='section-header'>빈 대여소 시간 분석</div>", unsafe_allow_html=True)
        
        # Station categories
        empty_stations = df[df['available_bikes'] == 0]
        nearly_empty_stations = df[(df['available_bikes'] > 0) & (df['available_bikes'] <= 2)]
        low_stations = df[(df['available_bikes'] > 2) & (df['available_bikes'] <= 5)]
        normal_stations = df[df['available_bikes'] > 5]
        
        # Display clickable cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(f"🔴 비어있음\n{len(empty_stations)}개", key="empty_btn", use_container_width=True):
                st.session_state.show_empty = not st.session_state.get('show_empty', False)
        
        with col2:
            if st.button(f"🟠 거의 없음\n{len(nearly_empty_stations)}개", key="nearly_btn", use_container_width=True):
                st.session_state.show_nearly = not st.session_state.get('show_nearly', False)
        
        with col3:
            if st.button(f"🟡 부족\n{len(low_stations)}개", key="low_btn", use_container_width=True):
                st.session_state.show_low = not st.session_state.get('show_low', False)
        
        with col4:
            if st.button(f"🟢 정상\n{len(normal_stations)}개", key="normal_btn", use_container_width=True):
                st.session_state.show_normal = not st.session_state.get('show_normal', False)
        
        # Show expanded station lists
        if st.session_state.get('show_empty', False) and len(empty_stations) > 0:
            with st.expander("비어있는 대여소 목록", expanded=True):
                display_stations = empty_stations[['station_name', 'station_id', 'station_capacity']].copy()
                display_stations.columns = ['대여소명', 'ID', '거치대']
                st.dataframe(display_stations, use_container_width=True, hide_index=True, height=400)
        
        if st.session_state.get('show_nearly', False) and len(nearly_empty_stations) > 0:
            with st.expander("거의 없는 대여소 목록", expanded=True):
                display_stations = nearly_empty_stations[['station_name', 'station_id', 'available_bikes', 'station_capacity']].copy()
                display_stations.columns = ['대여소명', 'ID', '자전거', '거치대']
                st.dataframe(display_stations, use_container_width=True, hide_index=True, height=400)
        
        if st.session_state.get('show_low', False) and len(low_stations) > 0:
            with st.expander("부족한 대여소 목록", expanded=True):
                display_stations = low_stations[['station_name', 'station_id', 'available_bikes', 'station_capacity']].copy()
                display_stations.columns = ['대여소명', 'ID', '자전거', '거치대']
                st.dataframe(display_stations, use_container_width=True, hide_index=True, height=400)
        
        if st.session_state.get('show_normal', False) and len(normal_stations) > 0:
            with st.expander("정상 대여소 목록", expanded=True):
                display_stations = normal_stations[['station_name', 'station_id', 'available_bikes', 'station_capacity']].copy()
                display_stations.columns = ['대여소명', 'ID', '자전거', '거치대']
                st.dataframe(display_stations.head(100), use_container_width=True, hide_index=True, height=400)
                if len(normal_stations) > 100:
                    st.info(f"전체 {len(normal_stations)}개 중 상위 100개만 표시")
        
        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs(["실시간 현황", "AI 예측 경보 (2시간 후)", "지도 보기", "데이터 관리"])
        
        with tab1:
            st.markdown("<div class='section-header'>대여소 실시간 현황</div>", unsafe_allow_html=True)
            
            # 현재 비어있거나 위험한 대여소 표시
            critical_stations = df[df['available_bikes'] <= 2].copy()
            critical_stations = critical_stations.sort_values('available_bikes')
            
            if len(critical_stations) > 0:
                # 구 정보 추출 (예시 함수)
                def extract_district(name):
                    for district in districts[1:]:  # '전체' 제외
                        if district in str(name):
                            return district
                    return '기타'
                
                critical_stations['구'] = critical_stations['station_name'].apply(extract_district)
                critical_stations['사용률'] = ((1 - critical_stations['available_bikes'] / critical_stations['station_capacity']) * 100).round(1)
                critical_stations['상태'] = critical_stations['available_bikes'].apply(
                    lambda x: '비어있음' if x == 0 else '거의없음'
                )
                
                # 테이블 표시
                display_df = critical_stations[['station_name', '구', 'available_bikes', 'station_capacity', '사용률', '상태']].copy()
                display_df.columns = ['대여소명', '구', '자전거', '거치대', '사용률(%)', '상태']
                
                # Display without styling to avoid theme conflicts
                st.dataframe(
                    display_df.head(20),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "상태": st.column_config.TextColumn(
                            "상태",
                            help="현재 자전거 상태"
                        ),
                        "자전거": st.column_config.NumberColumn(
                            "자전거",
                            format="%d"
                        ),
                        "거치대": st.column_config.NumberColumn(
                            "거치대", 
                            format="%d"
                        ),
                        "사용률(%)": st.column_config.NumberColumn(
                            "사용률(%)",
                            format="%.1f%%"
                        )
                    }
                )
                
                st.caption(f"총 {len(critical_stations)}개 대여소가 위험 상태입니다.")
            else:
                st.success("✅ 현재 모든 대여소가 정상 운영 중입니다.")
        
        with tab2:
            st.markdown("<div class='section-header'>AI 예측 경보 (2시간 후)</div>", unsafe_allow_html=True)
            
            # 현재 비어있지 않지만 2시간 후 예측되는 대여소만 분류
            not_empty_now = df[df['available_bikes'] > 0]
            
            # 위험도별 분류 (현재 비어있지 않은 대여소 중)
            predicted_stockout_stations = not_empty_now[not_empty_now['is_stockout_predicted'] == 1]
            high_risk = not_empty_now[(not_empty_now['stockout_probability'] >= 0.6) & 
                                      (not_empty_now['stockout_probability'] < 0.8)]
            medium_risk = not_empty_now[(not_empty_now['stockout_probability'] >= 0.4) & 
                                        (not_empty_now['stockout_probability'] < 0.6)]
            low_risk = not_empty_now[not_empty_now['stockout_probability'] < 0.4]
            
            # 현재 이미 비어있는 대여소
            already_empty = df[df['available_bikes'] == 0]
            
            # Clickable risk cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button(f"🟢 저위험\n{len(low_risk)}개\n<40%", key="low_risk_btn", use_container_width=True):
                    st.session_state.show_low_risk = not st.session_state.get('show_low_risk', False)
            
            with col2:
                if st.button(f"🟡 중위험\n{len(medium_risk)}개\n40-60%", key="medium_risk_btn", use_container_width=True):
                    st.session_state.show_medium_risk = not st.session_state.get('show_medium_risk', False)
            
            with col3:
                if st.button(f"🟠 고위험\n{len(high_risk)}개\n60-80%", key="high_risk_btn", use_container_width=True):
                    st.session_state.show_high_risk = not st.session_state.get('show_high_risk', False)
            
            with col4:
                if st.button(f"🔴 재고소진 예측\n{len(predicted_stockout_stations)}개\n2시간 내", key="stockout_btn", use_container_width=True):
                    st.session_state.show_stockout = not st.session_state.get('show_stockout', False)
            
            # Show expanded station lists for AI predictions
            if st.session_state.get('show_low_risk', False) and len(low_risk) > 0:
                with st.expander("저위험 대여소 목록 (<40%)", expanded=True):
                    display_low = low_risk[['station_name', 'station_id', 'available_bikes', 'station_capacity', 'stockout_probability']].copy()
                    display_low['stockout_probability'] = (display_low['stockout_probability'] * 100).round(1)
                    display_low.columns = ['대여소명', 'ID', '현재 자전거', '거치대', '2시간 후 확률(%)']
                    st.dataframe(display_low, use_container_width=True, hide_index=True, height=400)
            
            if st.session_state.get('show_medium_risk', False) and len(medium_risk) > 0:
                with st.expander("중위험 대여소 목록 (40-60%)", expanded=True):
                    display_medium = medium_risk[['station_name', 'station_id', 'available_bikes', 'station_capacity', 'stockout_probability']].copy()
                    display_medium['stockout_probability'] = (display_medium['stockout_probability'] * 100).round(1)
                    display_medium.columns = ['대여소명', 'ID', '현재 자전거', '거치대', '2시간 후 확률(%)']
                    st.dataframe(display_medium.sort_values('2시간 후 확률(%)', ascending=False), 
                                use_container_width=True, hide_index=True, height=400)
            
            if st.session_state.get('show_high_risk', False) and len(high_risk) > 0:
                with st.expander("고위험 대여소 목록 (60-80%)", expanded=True):
                    display_high = high_risk[['station_name', 'station_id', 'available_bikes', 'station_capacity', 'stockout_probability']].copy()
                    display_high['stockout_probability'] = (display_high['stockout_probability'] * 100).round(1)
                    display_high.columns = ['대여소명', 'ID', '현재 자전거', '거치대', '2시간 후 확률(%)']
                    st.dataframe(display_high.sort_values('2시간 후 확률(%)', ascending=False), 
                                use_container_width=True, hide_index=True, height=400)
            
            if st.session_state.get('show_stockout', False) and len(predicted_stockout_stations) > 0:
                with st.expander("재고소진 예측 대여소 목록 (2시간 내)", expanded=True):
                    display_stockout = predicted_stockout_stations[['station_name', 'station_id', 'available_bikes', 'station_capacity', 'stockout_probability']].copy()
                    display_stockout['stockout_probability'] = (display_stockout['stockout_probability'] * 100).round(1)
                    display_stockout.columns = ['대여소명', 'ID', '현재 자전거', '거치대', '2시간 후 확률(%)']
                    st.dataframe(display_stockout.sort_values('2시간 후 확률(%)', ascending=False), 
                                use_container_width=True, hide_index=True, height=400)
            
            # 고위험 대여소 테이블 (현재 비어있지 않지만 2시간 후 위험한 대여소만)
            risk_stations = not_empty_now[not_empty_now['stockout_probability'] >= 0.5]
            
            if len(risk_stations) > 0:
                st.markdown("#### 🔴 2시간 내 재고 소진 예측 대여소 (현재는 운영 중)")
                
                risk_df = risk_stations.copy()
                risk_df = risk_df.sort_values('stockout_probability', ascending=False)
                
                risk_df['구'] = risk_df['station_name'].apply(extract_district)
                risk_df['위험도'] = (risk_df['stockout_probability'] * 100).round(1)
                risk_df['상태'] = risk_df['stockout_probability'].apply(
                    lambda x: '재고소진 예측' if x >= 0.8 else '고위험' if x >= 0.6 else '중위험'
                )
                
                display_risk = risk_df[['station_name', '구', 'available_bikes', 'station_capacity', '위험도', '상태']].copy()
                display_risk.columns = ['대여소명', '구', '현재 자전거', '거치대', '2시간 후 확률(%)', '예측 상태']
                
                st.dataframe(
                    display_risk.head(20),
                    use_container_width=True,
                    hide_index=True
                )
                
                # 이미 비어있는 대여소 정보 추가
                if len(already_empty) > 0:
                    st.markdown("#### ⚠️ 현재 이미 비어있는 대여소")
                    st.info(f"현재 {len(already_empty)}개 대여소가 이미 재고가 소진된 상태입니다. (실시간 현황 탭 참조)")
            else:
                st.success("✅ 2시간 내 새로 재고 부족이 예상되는 대여소가 없습니다.")
        
        
        with tab3:
            st.markdown("<div class='section-header'>지도 보기</div>", unsafe_allow_html=True)
            
            # Check if we have coordinate data
            if 'latitude' in df.columns and 'longitude' in df.columns:
                # Filter options for map
                col1, col2 = st.columns([1, 3])
                with col1:
                    map_filter = st.selectbox(
                        "표시 기준",
                        ["현재 상태", "AI 예측 위험도"],
                        key="map_filter"
                    )
                
                # Create map centered on Seoul
                m = folium.Map(
                    location=[37.5665, 126.9780],  # Seoul coordinates
                    zoom_start=11,
                    tiles='OpenStreetMap'
                )
                
                # Add markers based on filter
                for idx, row in df.iterrows():
                    # Skip if no coordinates
                    if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                        continue
                    
                    # Determine color based on filter
                    if map_filter == "현재 상태":
                        if row['available_bikes'] == 0:
                            color = 'red'
                            status = '비어있음'
                        elif row['available_bikes'] <= 2:
                            color = 'orange'
                            status = '거의 없음'
                        elif row['available_bikes'] <= 5:
                            color = 'yellow'
                            status = '부족'
                        else:
                            color = 'green'
                            status = '정상'
                    else:  # AI 예측 위험도
                        prob = row.get('stockout_probability', 0)
                        if prob >= 0.8:
                            color = 'red'
                            status = f'매우위험 ({prob*100:.0f}%)'
                        elif prob >= 0.6:
                            color = 'orange'
                            status = f'고위험 ({prob*100:.0f}%)'
                        elif prob >= 0.4:
                            color = 'yellow'
                            status = f'중위험 ({prob*100:.0f}%)'
                        else:
                            color = 'green'
                            status = f'저위험 ({prob*100:.0f}%)'
                    
                    # Create popup text
                    popup_text = f"""
                    <b>{row['station_name']}</b><br>
                    ID: {row['station_id']}<br>
                    자전거: {row['available_bikes']}/{row['station_capacity']}<br>
                    상태: {status}<br>
                    <a href='https://maps.google.com/?q={row['latitude']},{row['longitude']}' target='_blank'>🗺️ Google Maps</a>
                    """
                    
                    # Add marker
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"{row['station_name']} ({row['available_bikes']}/{row['station_capacity']})",
                        icon=folium.Icon(color=color, icon='bicycle', prefix='fa')
                    ).add_to(m)
                
                # Display map
                st_folium(m, height=600, use_container_width=True)
                
                # Legend
                st.markdown("""
                <div style='display: flex; gap: 2rem; justify-content: center; margin-top: 1rem;'>
                    <span>🔴 비어있음/매우위험</span>
                    <span>🟠 거의없음/고위험</span>
                    <span>🟡 부족/중위험</span>
                    <span>🟢 정상/저위험</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("지도 데이터를 불러올 수 없습니다. API에서 좌표 정보를 확인해주세요.")
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 모델 정보")
                st.info("""
                **LightGBM 재고부족 예측 모델**
                - 예측 시점: 2시간 후
                - 정확도: 85.53%
                - ROC-AUC: 0.8955
                - F1-Score: 0.6177
                - 업데이트: 2025-08-19
                """)
            
            with col2:
                st.markdown("#### 🔧 시스템 상태")
                if st.button("시스템 상태 확인"):
                    try:
                        health = api_client.get_system_health()
                        if health and health.get('status') == 'healthy':
                            st.success(f"✅ 시스템 정상 작동")
                            st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            st.error("시스템 점검 필요")
                    except:
                        st.error("API 연결 실패")
        
        # 업데이트 시간 표시
        st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    else:
        st.error("데이터를 불러올 수 없습니다. API 서버를 확인해주세요.")
        st.info("API 서버가 실행 중인지 확인해주세요 (포트 8001)")
        
except Exception as e:
    st.error(f"오류 발생: {str(e)}")
    st.info("API 서버가 실행 중인지 확인해주세요")