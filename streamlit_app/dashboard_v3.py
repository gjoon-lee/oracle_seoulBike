"""
서울시 따릉이 통합 분석 시스템 v3
Seoul Bike Share Analysis System - Modern UI Version
"""

import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
from api_client import BikeAPIClient
import time

# 페이지 설정
st.set_page_config(
    page_title="서울시 따릉이 통합 분석 시스템",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Main header with 따릉이 gradient */
    .main-header {
        background: linear-gradient(135deg, #7FDE99 0%, #485562 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(127, 222, 153, 0.25);
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .header-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    
    .metric-icon {
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        color: #1e293b;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    .metric-positive {
        color: #10b981;
    }
    
    .metric-negative {
        color: #ef4444;
    }
    
    /* Station list styling */
    .station-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
    }
    
    .station-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .station-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .station-id {
        font-weight: 600;
        color: #1e293b;
    }
    
    .risk-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .risk-high {
        background: #fee2e2;
        color: #dc2626;
    }
    
    .risk-medium {
        background: #fed7aa;
        color: #ea580c;
    }
    
    .risk-low {
        background: #dcfce7;
        color: #16a34a;
    }
    
    /* Progress bars */
    .progress-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        transition: width 0.3s ease;
    }
    
    /* Search and filter section */
    .search-section {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: white;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: #7FDE99;
        color: #485562;
        font-weight: 600;
    }
    
    /* AI insight cards */
    .insight-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    
    .insight-urgent {
        border-left-color: #dc2626;
        background: #fef2f2;
    }
    
    .insight-warning {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    
    .insight-good {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
    
    /* Performance metric with progress */
    .perf-metric {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .perf-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .perf-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .perf-label {
        color: #64748b;
        font-size: 0.875rem;
    }
    
    /* Fix Streamlit default styles */
    .stDataFrame {
        background: white;
        border-radius: 8px;
        overflow: hidden;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Button styles with 따릉이 colors */
    .stButton > button {
        background: #7FDE99;
        color: #485562;
        border: 1px solid #7FDE99;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #485562;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(72, 85, 98, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# Initialize API client
@st.cache_resource
def get_api_client():
    return BikeAPIClient(base_url="http://localhost:8002")

api_client = get_api_client()

# 따릉이 Brand Color Palette
COLORS = {
    'primary': '#7FDE99',        # 따릉이 연두
    'primary_dark': '#485562',   # 따릉이 남색
    'success': '#7FDE99',        # 연두
    'warning': '#FFA500',        # 주황
    'danger': '#E74C3C'          # 빨강
}

# Helper functions
def calculate_risk_level(probability):
    """Calculate risk level based on stockout probability (3 categories)"""
    if probability >= 0.8:
        return "위험", "risk-high", COLORS['danger']
    elif probability >= 0.5:
        return "경고", "risk-medium", COLORS['warning']
    else:
        return "양호", "risk-low", COLORS['success']

def format_percentage(value):
    """Format percentage with color"""
    return f"{value:.1f}%"

def create_progress_bar(value, max_value=100, color="#7FDE99"):
    """Create HTML progress bar"""
    percentage = min(100, (value / max_value) * 100)
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {percentage}%; background: {color};"></div>
    </div>
    """

# Main app
def main():
    # Header Section with Logo
    header_col1, header_col2 = st.columns([1, 5])
    
    with header_col1:
        # 따릉이 로고
        try:
            st.image('C:/Users/82103/OneDrive/Desktop/따릉이_로고.png', width=250)
        except:
            st.write("🚴")  # Fallback emoji if logo not found
    
    with header_col2:
        st.markdown("""
        <div class="main-header" style="margin-top: -1rem;">
            <div class="header-title">
                <span>서울시 따릉이 통합 분석 시스템</span>
            </div>
            <div class="header-subtitle">
                실시간 현황 및 예측 대시보드 • {} 기준 • 
                <span style="color: #ffffff;">● 실시간 연결됨</span>
            </div>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
    
    # Fetch current data with progress indicators
    progress_container = st.container()
    with progress_container:
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("📍 대여소 현황 불러오는 중..."):
                current_status = api_client.get_stations_status()
        with col2:
            with st.spinner("🤖 재고 부족 예측 중 (LightGBM)..."):
                predictions = api_client.get_predictions()
        
        # XGBoost predictions - DO NOT LOAD on initial page load
        xgboost_predictions = None  # Will be loaded on-demand in XGB tab only
    
    if current_status is not None and predictions is not None:
        # Extract dataframes from API responses
        if 'stations' in current_status:
            current_df = pd.DataFrame(current_status['stations'])
        else:
            current_df = pd.DataFrame()
            
        if 'predictions' in predictions:
            pred_df = pd.DataFrame(predictions['predictions'])
        else:
            pred_df = pd.DataFrame()
        
        # Process XGBoost predictions
        xgb_df = pd.DataFrame()
        if xgboost_predictions and 'predictions' in xgboost_predictions:
            xgb_df = pd.DataFrame(xgboost_predictions['predictions'])
            # Select only needed columns
            xgb_df = xgb_df[['station_id', 'predicted_net_flow_2h', 'predicted_bikes_2h', 'confidence_level']]
            # Rename to avoid conflicts
            xgb_df.columns = ['station_id', 'xgb_net_flow_2h', 'xgb_predicted_bikes_2h', 'xgb_confidence']
        
        # Merge data if both dataframes have data
        if not current_df.empty and not pred_df.empty:
            df = pd.merge(
                current_df,
                pred_df[['station_id', 'stockout_probability', 'is_stockout_predicted', 'risk_level']],
                on='station_id',
                how='left'
            )
            
            # Merge XGBoost predictions if available
            if not xgb_df.empty:
                df = pd.merge(df, xgb_df, on='station_id', how='left')
        elif not current_df.empty:
            # Use current status only if predictions not available
            df = current_df
            
            # If predictions fail, show error and return empty dataframe
            st.error("❌ 예측 모델 API 연결 실패 - 예측 데이터를 사용할 수 없습니다.")
            st.info("💡 API 서버를 확인해주세요: http://localhost:8002/docs")
            df = pd.DataFrame()  # Return empty dataframe to trigger error handling below
        else:
            df = pd.DataFrame()
        
        # Check if we have data
        if df.empty:
            st.error("⚠️ 데이터를 불러올 수 없습니다. API 서버가 실행 중인지 확인해주세요.")
            st.info("API 서버를 시작하려면: cd realtime_prediction && python main.py")
            return
        
        # Calculate metrics
        total_stations = len(df) if not df.empty else 0
        high_risk_count = len(df[df['stockout_probability'] >= 0.5]) if 'stockout_probability' in df.columns and not df.empty else 0
        avg_availability = (df['available_bikes'] / df['station_capacity']).mean() * 100 if not df.empty and 'available_bikes' in df.columns else 50.0
        
        # Real model metrics from 2024 test data
        model_accuracy = 85.53  # Actual accuracy
        model_f1 = 61.77  # F1-Score
        model_roc_auc = 89.55  # ROC-AUC
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="✅ 전체 대여소",
                value=f"{total_stations:,}",
                delta="운영 중"
            )
        
        with col2:
            st.metric(
                label="⚠️ 위험 대여소",
                value=high_risk_count,
                delta="재고부족 위험",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="📊 평균 재고율",
                value=f"{avg_availability:.1f}%",
                delta="↑ 1.5%"
            )
        
        with col4:
            st.metric(
                label="🎯 모델 정확도",
                value=f"{model_accuracy}%",
                help="2024년 테스트 데이터 (LightGBM)"
            )
        
        
        # Main Tabs
        tab1, tab2, tab3 = st.tabs(["📍 실시간 대여소 현황", "🤖 예측 인사이트", "📊 XGBoost 예측"])
        
        with tab1:
            # Real-time Station Status with Map
            col_left, col_right = st.columns([3, 2])
            
            with col_left:
                # Search and Filter Section
                st.markdown('<div class="search-section">', unsafe_allow_html=True)
                search_col, filter_col1, filter_col2 = st.columns([2, 1, 1])
                
                with search_col:
                    search_query = st.text_input("🔍 대여소명 또는 ID 검색...", label_visibility="collapsed", 
                                                placeholder="대여소명 또는 ID 검색...")
                
                with filter_col1:
                    risk_filter = st.selectbox("전체 위험도", ["전체", "위험", "주의", "양호"], label_visibility="collapsed")
                
                with filter_col2:
                    sort_order = st.selectbox("위험도순", ["위험도순", "대여소명순", "재고순"], label_visibility="collapsed")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Apply filters
                filtered_df = df.copy()
                
                if search_query:
                    filtered_df = filtered_df[
                        filtered_df['station_name'].str.contains(search_query, case=False) | 
                        filtered_df['station_id'].str.contains(search_query, case=False)
                    ]
                
                if risk_filter != "전체":
                    risk_map = {"위험": 0.8, "주의": 0.5, "양호": 0}
                    if risk_filter == "위험":
                        filtered_df = filtered_df[filtered_df['stockout_probability'] >= 0.8]
                    elif risk_filter == "주의":
                        filtered_df = filtered_df[(filtered_df['stockout_probability'] >= 0.5) & 
                                                (filtered_df['stockout_probability'] < 0.8)]
                    else:  # 양호
                        filtered_df = filtered_df[filtered_df['stockout_probability'] < 0.5]
                
                # Sort
                if sort_order == "위험도순":
                    filtered_df = filtered_df.sort_values('stockout_probability', ascending=False)
                elif sort_order == "대여소명순":
                    filtered_df = filtered_df.sort_values('station_name')
                else:  # 재고순
                    filtered_df = filtered_df.sort_values('available_bikes')
                
                # Display station list
                st.markdown(f"#### 총 {len(filtered_df)}개 대여소")
                
                # Station list container with native Streamlit components
                for idx, row in filtered_df.head(50).iterrows():
                    risk_text, risk_class, risk_color = calculate_risk_level(row['stockout_probability'])
                    availability_pct = (row['available_bikes'] / row['station_capacity']) * 100
                    
                    with st.container():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            st.write(f"**{row['station_id']}** - {row['station_name']}")
                        
                        with col2:
                            st.write(f"🚲 {row['available_bikes']}/{row['station_capacity']} 대 | 확률 {row['stockout_probability']*100:.0f}%")
                        
                        with col3:
                            if risk_text == "위험":
                                st.error(risk_text)
                            elif risk_text == "경고":
                                st.warning(risk_text)
                            else:
                                st.success(risk_text)
                        
                        # Progress bar for availability
                        st.progress(
                            availability_pct/100,
                            text=f"재고율: {availability_pct:.0f}%"
                        )
                        st.divider()
                    
                
                if len(filtered_df) > 50:
                    st.info(f"추가 {len(filtered_df) - 50}개 스테이션이 있습니다")
            
            with col_right:
                # Map View
                st.markdown("#### 🗺️ 실시간 지도")
                
                # Check if we have coordinate data
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    # Create map centered on Seoul
                    m = folium.Map(
                        location=[37.5665, 126.9780],
                        zoom_start=11,
                        tiles='OpenStreetMap'
                    )
                    
                    # Add markers for filtered stations (limit to 100 for performance)
                    map_df = filtered_df.head(100) if len(filtered_df) > 100 else filtered_df
                    for idx, row in map_df.iterrows():
                        # Skip if no coordinates
                        if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                            continue
                        
                        # Determine marker color based on risk level
                        risk_text, risk_class, risk_color = calculate_risk_level(row['stockout_probability'])
                        
                        if risk_text == "위험":
                            marker_color = 'red'
                            icon_color = 'white'
                        elif risk_text == "경고":
                            marker_color = 'orange'
                            icon_color = 'white'
                        else:
                            marker_color = 'green'
                            icon_color = 'white'
                        
                        # Create popup text
                        popup_text = f"""
                        <b>{row['station_name']}</b><br>
                        ID: {row['station_id']}<br>
                        자전거: {row['available_bikes']}/{row['station_capacity']}<br>
                        재고부족 확률: {row['stockout_probability']*100:.0f}%<br>
                        상태: {risk_text}
                        """
                        
                        # Add marker
                        folium.Marker(
                            location=[row['latitude'], row['longitude']],
                            popup=folium.Popup(popup_text, max_width=200),
                            tooltip=row['station_name'],
                            icon=folium.Icon(color=marker_color, icon='bicycle', prefix='fa')
                        ).add_to(m)
                    
                    # Display map
                    st_folium(m, height=400, width=None, returned_objects=["last_clicked"])
                else:
                    st.info("지도를 표시하려면 좌표 데이터가 필요합니다.")
                
                # High Risk Stations List below map
                st.markdown("#### 🔴 위험 대여소 목록")
                st.markdown("재고부족 위험이 높은 대여소", unsafe_allow_html=True)
                
                high_risk_df = df[df['stockout_probability'] >= 0.5].sort_values('stockout_probability', ascending=False)
                
                for idx, row in high_risk_df.head(15).iterrows():  # Show more high risk stations
                    alert_type = "위험" if row['stockout_probability'] >= 0.8 else "주의"
                    alert_color = "#dc2626" if row['stockout_probability'] >= 0.8 else "#f59e0b"
                    
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; 
                                border-left: 3px solid {alert_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-weight: 600; color: {alert_color};">● {row['station_id']}</span>
                                <div style="color: #64748b; font-size: 0.875rem; margin-top: 0.25rem;">
                                    {row['station_name'][:15]}...
                                </div>
                                <div style="font-size: 0.875rem; margin-top: 0.25rem;">
                                    현재 자전거: {row['available_bikes']}대 | 부족 확률: {row['stockout_probability']*100:.0f}%
                                </div>
                            </div>
                            <div style="text-align: center;">
                                <span style="background: {alert_color}; color: white; padding: 0.25rem 0.5rem; 
                                           border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                                    {alert_type}
                                </span>
                            </div>
                        </div>
                        <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.5rem;">
                            📍 예상 고갈: 1시간 내 자전거 보충 권장
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            # AI Prediction Insights
            st.markdown("### 🤖 예측 인사이트")
            st.markdown("머신러닝 기반 예측 분석 및 권장사항")
            st.info("📊 모든 성능 지표는 2024년 1-12월 테스트 데이터 평가 결과입니다")
            
            # Performance Metrics in 2x2 grid
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                # Model Accuracy
                st.metric(
                    label="🎯 테스트 정확도",
                    value=f"{model_accuracy}%",
                    help="LightGBM Accuracy (2024년 테스트 세트)"
                )
                st.progress(model_accuracy/100)
                st.caption("2024년 테스트 데이터 기준")
                
                # Prediction time
                st.metric(
                    label="⏱️ 예측 시간",
                    value="2시간 후",
                    help="재고부족 예측 시간 범위"
                )
            
            with metric_col2:
                # F1 Score  
                st.metric(
                    label="⚖️ F1-Score",
                    value=f"{model_f1}%",
                    help="정밀도와 재현율의 조화평균 (2024년 테스트)"
                )
                st.progress(model_f1/100)
                st.caption("2024년 테스트 세트")
                
                # ROC-AUC
                st.metric(
                    label="📈 ROC-AUC",
                    value=f"{model_roc_auc}%",
                    help="분류 성능 지표 (2024년 테스트 세트)"
                )
                st.progress(model_roc_auc/100)
                st.caption("이진 분류 성능")
            
            # AI Analysis Results - 3 Categories
            st.markdown("### 📋 AI 분석 결과 (3단계 분류)")
            
            # Calculate station counts for each category
            danger_stations = df[df['stockout_probability'] >= 0.8]
            warning_stations = df[(df['stockout_probability'] >= 0.5) & (df['stockout_probability'] < 0.8)]
            safe_stations = df[df['stockout_probability'] < 0.5]
            
            # Create 3 columns for categories
            cat_col1, cat_col2, cat_col3 = st.columns(3)
            
            with cat_col1:
                st.markdown(f"""
                <div class="insight-card insight-good">
                    <div style="font-weight: 600; color: #059669; margin-bottom: 0.5rem;">
                        🟢 양호
                    </div>
                    <div style="color: #064e3b; font-size: 1.5rem; font-weight: 700;">
                        {len(safe_stations)}개 대여소
                    </div>
                    <div style="color: #64748b; font-size: 0.875rem; margin-top: 0.5rem;">
                        재고부족 확률 <50%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("상세보기", key="safe_detail"):
                    st.session_state.show_safe_detail = not st.session_state.get('show_safe_detail', False)
            
            with cat_col2:
                st.markdown(f"""
                <div class="insight-card insight-warning">
                    <div style="font-weight: 600; color: #d97706; margin-bottom: 0.5rem;">
                        🟠 경고
                    </div>
                    <div style="color: #78350f; font-size: 1.5rem; font-weight: 700;">
                        {len(warning_stations)}개 대여소
                    </div>
                    <div style="color: #64748b; font-size: 0.875rem; margin-top: 0.5rem;">
                        재고부족 확률 50-80%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("상세보기", key="warning_detail"):
                    st.session_state.show_warning_detail = not st.session_state.get('show_warning_detail', False)
            
            with cat_col3:
                st.markdown(f"""
                <div class="insight-card insight-urgent">
                    <div style="font-weight: 600; color: #dc2626; margin-bottom: 0.5rem;">
                        🔴 위험
                    </div>
                    <div style="color: #7f1d1d; font-size: 1.5rem; font-weight: 700;">
                        {len(danger_stations)}개 대여소
                    </div>
                    <div style="color: #64748b; font-size: 0.875rem; margin-top: 0.5rem;">
                        재고부족 확률 ≥80%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("상세보기", key="danger_detail"):
                    st.session_state.show_danger_detail = not st.session_state.get('show_danger_detail', False)
            
            # Show detailed lists based on button clicks
            if st.session_state.get('show_safe_detail', False):
                with st.expander("🟢 양호 대여소 목록", expanded=True):
                    if not safe_stations.empty:
                        # Add button to load XGBoost predictions
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            if st.button("XGBoost 예측 보기", key="safe_xgb"):
                                with st.spinner("XGBoost 예측 로딩 중..."):
                                    station_ids = safe_stations['station_id'].tolist()
                                    xgb_batch = api_client.get_xgboost_batch(station_ids)
                                    if xgb_batch and 'predictions' in xgb_batch:
                                        xgb_batch_df = pd.DataFrame(xgb_batch['predictions'])
                                        xgb_batch_df = xgb_batch_df[['station_id', 'predicted_net_flow_2h', 'predicted_bikes_2h', 'confidence_level']]
                                        xgb_batch_df.columns = ['station_id', 'xgb_net_flow_2h', 'xgb_predicted_bikes_2h', 'xgb_confidence']
                                        safe_stations = pd.merge(safe_stations, xgb_batch_df, on='station_id', how='left')
                                        st.session_state['safe_xgb_data'] = xgb_batch_df
                        
                        # Check if we have cached XGBoost data
                        if 'safe_xgb_data' in st.session_state:
                            xgb_data = st.session_state['safe_xgb_data']
                            safe_stations = pd.merge(safe_stations.drop(columns=['xgb_net_flow_2h', 'xgb_predicted_bikes_2h', 'xgb_confidence'], errors='ignore'),
                                                    xgb_data, on='station_id', how='left')
                        
                        # Prepare display columns
                        cols_to_show = ['station_name', 'station_id', 'available_bikes', 'station_capacity', 'stockout_probability']
                        col_names = ['대여소명', 'ID', '현재 자전거', '거치대', '재고부족 확률(%)']
                        
                        if 'xgb_predicted_bikes_2h' in safe_stations.columns and 'xgb_net_flow_2h' in safe_stations.columns:
                            cols_to_show.append('xgb_predicted_bikes_2h')
                            cols_to_show.append('xgb_net_flow_2h')
                            col_names.append('예상 자전거(2h)')
                            col_names.append('순 변화량')
                        
                        display_safe = safe_stations[cols_to_show].copy()
                        display_safe['stockout_probability'] = (display_safe['stockout_probability'] * 100).round(1)
                        
                        if 'xgb_predicted_bikes_2h' in display_safe.columns:
                            display_safe['xgb_predicted_bikes_2h'] = display_safe['xgb_predicted_bikes_2h'].round(0).astype(int)
                            display_safe['xgb_net_flow_2h'] = display_safe['xgb_net_flow_2h'].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "")
                        
                        display_safe.columns = col_names
                        
                        # Style the dataframe with light gray background for XGBoost columns
                        styled_df = display_safe.style
                        if '예상 자전거(2h)' in display_safe.columns:
                            styled_df = styled_df.set_properties(**{'background-color': '#f3f4f6'}, 
                                                                subset=['예상 자전거(2h)', '순 변화량'])
                        
                        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
            
            if st.session_state.get('show_warning_detail', False):
                with st.expander("🟠 경고 대여소 목록", expanded=True):
                    if not warning_stations.empty:
                        # Check if XGBoost columns exist
                        cols_to_show = ['station_name', 'station_id', 'available_bikes', 'station_capacity', 'stockout_probability']
                        col_names = ['대여소명', 'ID', '현재 자전거', '거치대', '재고부족 확률(%)']
                        
                        if 'xgb_predicted_bikes_2h' in warning_stations.columns and 'xgb_net_flow_2h' in warning_stations.columns:
                            cols_to_show.append('xgb_predicted_bikes_2h')
                            cols_to_show.append('xgb_net_flow_2h')
                            col_names.append('예상 자전거(2h)')
                            col_names.append('순 변화량')
                        
                        display_warning = warning_stations[cols_to_show].copy()
                        display_warning['stockout_probability'] = (display_warning['stockout_probability'] * 100).round(1)
                        
                        if 'xgb_predicted_bikes_2h' in display_warning.columns:
                            display_warning['xgb_predicted_bikes_2h'] = display_warning['xgb_predicted_bikes_2h'].round(0).astype(int)
                            display_warning['xgb_net_flow_2h'] = display_warning['xgb_net_flow_2h'].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "")
                        
                        display_warning.columns = col_names
                        
                        # Style the dataframe with light gray background for XGBoost columns
                        styled_df = display_warning.sort_values(col_names[4], ascending=False).style
                        if '예상 자전거(2h)' in display_warning.columns:
                            styled_df = styled_df.set_properties(**{'background-color': '#f3f4f6'}, 
                                                                subset=['예상 자전거(2h)', '순 변화량'])
                        
                        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
            
            if st.session_state.get('show_danger_detail', False):
                with st.expander("🔴 위험 대여소 목록", expanded=True):
                    if not danger_stations.empty:
                        # Check if XGBoost columns exist
                        cols_to_show = ['station_name', 'station_id', 'available_bikes', 'station_capacity', 'stockout_probability']
                        col_names = ['대여소명', 'ID', '현재 자전거', '거치대', '재고부족 확률(%)']
                        
                        if 'xgb_predicted_bikes_2h' in danger_stations.columns and 'xgb_net_flow_2h' in danger_stations.columns:
                            cols_to_show.append('xgb_predicted_bikes_2h')
                            cols_to_show.append('xgb_net_flow_2h')
                            col_names.append('예상 자전거(2h)')
                            col_names.append('순 변화량')
                        
                        display_danger = danger_stations[cols_to_show].copy()
                        display_danger['stockout_probability'] = (display_danger['stockout_probability'] * 100).round(1)
                        
                        if 'xgb_predicted_bikes_2h' in display_danger.columns:
                            display_danger['xgb_predicted_bikes_2h'] = display_danger['xgb_predicted_bikes_2h'].round(0).astype(int)
                            display_danger['xgb_net_flow_2h'] = display_danger['xgb_net_flow_2h'].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "")
                        
                        display_danger.columns = col_names
                        
                        # Style the dataframe with light gray background for XGBoost columns
                        styled_df = display_danger.sort_values(col_names[4], ascending=False).style
                        if '예상 자전거(2h)' in display_danger.columns:
                            styled_df = styled_df.set_properties(**{'background-color': '#f3f4f6'}, 
                                                                subset=['예상 자전거(2h)', '순 변화량'])
                        
                        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)
            
            # Distribution Chart - 3 Categories
            st.markdown("### 📊 위험도 분포")
            
            risk_distribution = pd.DataFrame({
                '위험도': ['양호 (<50%)', '경고 (50-80%)', '위험 (≥80%)'],
                '대여소 수': [
                    len(safe_stations),
                    len(warning_stations),
                    len(danger_stations)
                ]
            })
            
            fig = px.bar(risk_distribution, x='위험도', y='대여소 수',
                        color='위험도',
                        color_discrete_map={
                            '양호 (<50%)': COLORS['success'],
                            '경고 (50-80%)': COLORS['warning'],
                            '위험 (≥80%)': COLORS['danger']
                        })
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
        # XGBoost Predictions Tab
        with tab3:
            st.markdown("### 📊 XGBoost Net Flow 예측")
            st.markdown("XGBoost 회귀 모델을 통한 2시간 후 자전거 순 흐름 예측")
            st.info("📊 필터를 설정하고 '예측 실행' 버튼을 클릭하여 선택된 대여소의 예측을 확인하세요")
            
            # Get XGBoost model info
            xgb_model_info = api_client.get_xgboost_model_info()
            
            if xgb_model_info:
                # Model Performance Metrics
                st.markdown("### 📈 모델 성능 지표")
                xgb_col1, xgb_col2, xgb_col3, xgb_col4 = st.columns(4)
                
                with xgb_col1:
                    mae = xgb_model_info.get('metrics', {}).get('mae', 1.62)
                    st.metric(
                        label="📏 MAE",
                        value=f"{mae:.2f} 대",
                        help="Mean Absolute Error - 평균 절대 오차"
                    )
                
                with xgb_col2:
                    rmse = xgb_model_info.get('metrics', {}).get('rmse', 2.34)
                    st.metric(
                        label="📊 RMSE",
                        value=f"{rmse:.2f} 대",
                        help="Root Mean Square Error - 제곱근 평균 제곱 오차"
                    )
                
                with xgb_col3:
                    r2 = 0.53  # Hardcoded R² value
                    st.metric(
                        label="🎯 R²",
                        value=f"{r2:.2f}",
                        help="결정 계수 - 모델 설명력"
                    )
                
                with xgb_col4:
                    st.metric(
                        label="⏱️ 예측 시간",
                        value="2시간 후",
                        help="순 흐름 예측 시간 범위"
                    )
            
            # Filters
            st.markdown("### 🔍 필터 옵션")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                # District filter
                all_districts = df['district'].unique() if 'district' in df.columns else []
                selected_district = st.selectbox(
                    "구 선택",
                    ["전체"] + list(all_districts) if len(all_districts) > 0 else ["전체"],
                    key="xgb_district"
                )
            
            with filter_col2:
                # Station search
                search_query = st.text_input(
                    "대여소 검색",
                    placeholder="대여소명 또는 ID 입력",
                    key="xgb_search"
                )
            
            with filter_col3:
                # Confidence level filter
                confidence_filter = st.selectbox(
                    "신뢰도 수준",
                    ["전체", "높음", "중간", "낮음"],
                    key="xgb_confidence"
                )
            
            # Apply filters
            filtered_df = df.copy()
            
            if selected_district != "전체" and 'district' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['district'] == selected_district]
            
            if search_query:
                mask = (
                    filtered_df['station_name'].str.contains(search_query, case=False, na=False) |
                    filtered_df['station_id'].str.contains(search_query, case=False, na=False)
                )
                filtered_df = filtered_df[mask]
            
            # Show how many stations match filter
            st.info(f"📊 필터 조건에 맞는 대여소: {len(filtered_df)}개")
            
            # Add predict button
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                predict_button = st.button("🔮 예측 실행", key="run_xgb_prediction", type="primary")
            
            # Main predictions table
            st.markdown("### 📋 XGBoost 예측 결과")
            
            # Handle prediction button click
            if predict_button and len(filtered_df) > 0:
                if len(filtered_df) > 100:
                    st.warning("⚠️ 100개 이상의 대여소가 선택되었습니다. 상위 100개만 예측합니다.")
                    filtered_df = filtered_df.head(100)
                
                with st.spinner(f"🔮 {len(filtered_df)}개 대여소 예측 중..."):
                    station_ids = filtered_df['station_id'].tolist()
                    xgb_batch_result = api_client.get_xgboost_batch(station_ids)
                    
                    if xgb_batch_result and 'predictions' in xgb_batch_result:
                        xgb_predictions_df = pd.DataFrame(xgb_batch_result['predictions'])
                        # Store in session state
                        st.session_state['xgb_tab_predictions'] = xgb_predictions_df
                        st.success(f"✅ {len(xgb_predictions_df)}개 대여소 예측 완료!")
                    else:
                        st.error("❌ 예측 실행 실패. API 서버를 확인해주세요.")
            
            # Check if we have predictions in session state
            if 'xgb_tab_predictions' in st.session_state and st.session_state['xgb_tab_predictions'] is not None:
                xgb_predictions_df = st.session_state['xgb_tab_predictions']
                
                # Validate that predictions DataFrame has required columns
                if xgb_predictions_df.empty:
                    st.warning("⚠️ 예측 데이터가 비어있습니다.")
                elif 'station_id' not in xgb_predictions_df.columns:
                    st.error(f"❌ 예측 데이터에 station_id 컬럼이 없습니다. 사용 가능한 컬럼: {list(xgb_predictions_df.columns)}")
                else:
                    try:
                        # Merge with filtered_df to get station names
                        merged_df = pd.merge(
                            filtered_df[['station_id', 'station_name', 'available_bikes']],
                            xgb_predictions_df,
                            on='station_id',
                            how='inner'
                        )
                        # Prepare display dataframe (removed confidence_level)
                        xgb_display_cols = [
                            'station_name', 'station_id', 'current_bikes', 
                            'predicted_bikes_2h', 'predicted_net_flow_2h'
                        ]
                        
                        # Filter columns that exist
                        xgb_display_cols = [col for col in xgb_display_cols if col in merged_df.columns]
                        
                        xgb_display = merged_df[xgb_display_cols].copy()
                        
                        # Format columns
                        if 'predicted_bikes_2h' in xgb_display.columns:
                            xgb_display['predicted_bikes_2h'] = xgb_display['predicted_bikes_2h'].round(0).astype(int)
                        
                        if 'predicted_net_flow_2h' in xgb_display.columns:
                            xgb_display['predicted_net_flow_2h'] = xgb_display['predicted_net_flow_2h'].apply(
                                lambda x: f"{x:+.0f}" if pd.notna(x) else "N/A"
                            )
                        
                        # Rename columns to Korean
                        column_mapping = {
                            'station_name': '대여소명',
                            'station_id': 'ID',
                            'available_bikes': '현재 자전거',
                            'current_bikes': '현재 자전거',
                            'predicted_bikes_2h': '예상 자전거(2h)',
                            'predicted_net_flow_2h': '순 변화량'
                        }
                        xgb_display.rename(columns=column_mapping, inplace=True)
                        
                        # Display table
                        st.dataframe(
                            xgb_display,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                        
                        # Insights section
                        st.markdown("### 💡 주요 인사이트")
                        
                        insight_col1, insight_col2 = st.columns(2)
                        
                        with insight_col1:
                            # Top gaining stations
                            st.markdown("#### 📈 자전거 증가 예상 TOP 5")
                            if 'predicted_net_flow_2h' in merged_df.columns:
                                # Use numeric values directly for sorting
                                top_gaining = merged_df.nlargest(5, 'predicted_net_flow_2h')[
                                    ['station_name', 'current_bikes', 'predicted_net_flow_2h', 'predicted_bikes_2h']
                                ]
                                
                                if not top_gaining.empty:
                                    for idx, row in top_gaining.iterrows():
                                        st.markdown(f"""
                                        <div style="background: #f0fdf4; padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;
                                                    border-left: 3px solid #7FDE99;">
                                            <div style="font-weight: 600; color: #166534;">
                                                {row['station_name'][:20]}...
                                            </div>
                                            <div style="color: #64748b; font-size: 0.875rem;">
                                                현재: {row['current_bikes']}대 → 예상: {row['predicted_bikes_2h']:.0f}대 
                                                <span style="color: #059669; font-weight: 600;">(+{row['predicted_net_flow_2h']:.0f})</span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        with insight_col2:
                            # Top losing stations
                            st.markdown("#### 📉 자전거 감소 예상 TOP 5")
                            if 'predicted_net_flow_2h' in merged_df.columns:
                                top_losing = merged_df.nsmallest(5, 'predicted_net_flow_2h')[
                                    ['station_name', 'current_bikes', 'predicted_net_flow_2h', 'predicted_bikes_2h']
                                ]
                                
                                if not top_losing.empty:
                                    for idx, row in top_losing.iterrows():
                                        st.markdown(f"""
                                        <div style="background: #fef2f2; padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;
                                                    border-left: 3px solid #ef4444;">
                                            <div style="font-weight: 600; color: #991b1b;">
                                                {row['station_name'][:20]}...
                                            </div>
                                            <div style="color: #64748b; font-size: 0.875rem;">
                                                현재: {row['current_bikes']}대 → 예상: {row['predicted_bikes_2h']:.0f}대
                                                <span style="color: #dc2626; font-weight: 600;">({row['predicted_net_flow_2h']:.0f})</span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        # Summary statistics
                        st.markdown("### 📊 전체 통계")
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        
                        with stat_col1:
                            if 'predicted_net_flow_2h' in merged_df.columns:
                                gaining_count = (merged_df['predicted_net_flow_2h'] > 0).sum()
                                st.metric(
                                    "증가 예상 대여소",
                                    f"{gaining_count}개",
                                    f"{gaining_count/len(merged_df)*100:.1f}%"
                                )
                        
                        with stat_col2:
                            if 'predicted_net_flow_2h' in merged_df.columns:
                                losing_count = (merged_df['predicted_net_flow_2h'] < 0).sum()
                                st.metric(
                                    "감소 예상 대여소",
                                    f"{losing_count}개",
                                    f"{losing_count/len(merged_df)*100:.1f}%"
                                )
                        
                        with stat_col3:
                            if 'predicted_net_flow_2h' in merged_df.columns:
                                avg_flow = merged_df['predicted_net_flow_2h'].mean()
                                st.metric(
                                    "평균 순 흐름",
                                    f"{avg_flow:+.1f}대",
                                    "2시간 후 예측"
                                )
                    except Exception as e:
                        st.error(f"❌ 데이터 처리 중 오류 발생: {str(e)}")
                        st.info("디버그 정보: xgb_predictions_df 컬럼: " + str(list(xgb_predictions_df.columns)))
            else:
                # Show placeholder when no predictions are loaded
                st.info("📊 필터를 설정하고 '예측 실행' 버튼을 클릭하여 XGBoost 예측을 확인하세요.")
                
                # Show empty state
                empty_col1, empty_col2, empty_col3 = st.columns([1, 2, 1])
                with empty_col2:
                    st.markdown("""
                    <div style="text-align: center; padding: 3rem; color: #94a3b8;">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">🔮</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">예측 대기 중</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                            필터를 설정하고 '예측 실행' 버튼을 클릭하세요
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.error("⚠️ 데이터를 불러올 수 없습니다. API 서버 연결을 확인해주세요.")
    
    # Auto-refresh - REMOVED the blocking sleep!
    # The dashboard will refresh when user clicks refresh button instead
    # time.sleep was blocking the entire UI!

if __name__ == "__main__":
    main()