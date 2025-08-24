"""
Reusable UI components for Streamlit dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import numpy as np

def create_metric_card(title: str, value: str, delta: Optional[str] = None, 
                       delta_color: str = "normal") -> None:
    """Create a metric card component"""
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)

def create_status_table(df: pd.DataFrame, title: str = "대여소 현황", 
                        highlight_stockout: bool = True) -> None:
    """Create a formatted status table"""
    
    st.subheader(title)
    
    if highlight_stockout and 'is_stockout' in df.columns:
        # Apply conditional formatting
        def highlight_empty(row):
            if row.get('is_stockout', 0) == 1:
                return ['background-color: #ffebee'] * len(row)
            elif row.get('utilization_rate', 0) > 0.8:
                return ['background-color: #fff3e0'] * len(row)
            return [''] * len(row)
        
        styled_df = df.style.apply(highlight_empty, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

def create_hourly_chart(data: pd.DataFrame, station_id: str = "전체") -> go.Figure:
    """Create hourly demand chart"""
    
    fig = go.Figure()
    
    # Add demand line
    if 'demand' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['hour'],
            y=data['demand'],
            mode='lines+markers',
            name='예측 수요',
            line=dict(color='#5B47FB', width=3),
            marker=dict(size=8)
        ))
    
    # Add temperature line if available
    if 'temperature' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['hour'],
            y=data['temperature'],
            mode='lines',
            name='온도 (°C)',
            line=dict(color='#FF9800', width=2, dash='dot'),
            yaxis='y2'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{station_id} - 시간별 예측",
        xaxis=dict(
            title="시간",
            tickmode='linear',
            tick0=0,
            dtick=3
        ),
        yaxis=dict(
            title="예측 수요",
            side='left'
        ),
        yaxis2=dict(
            title="온도 (°C)",
            side='right',
            overlaying='y'
        ),
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_stockout_prediction_table(predictions: List[Dict]) -> pd.DataFrame:
    """Create stockout prediction table with risk levels"""
    
    df = pd.DataFrame(predictions)
    
    # Add risk level
    def get_risk_level(prob):
        if prob >= 0.8:
            return "🔴 매우 위험"
        elif prob >= 0.6:
            return "🟠 위험"
        elif prob >= 0.4:
            return "🟡 주의"
        else:
            return "🟢 양호"
    
    if 'stockout_probability' in df.columns:
        df['위험도'] = df['stockout_probability'].apply(get_risk_level)
    
    # Format columns
    display_columns = {
        'station_id': '대여소 ID',
        'station_name': '대여소명',
        'current_available_bikes': '현재 자전거',
        'stockout_probability': '부족 확률',
        '위험도': '위험도'
    }
    
    # Select and rename columns
    available_cols = [col for col in display_columns.keys() if col in df.columns]
    df = df[available_cols]
    df.columns = [display_columns[col] for col in available_cols]
    
    # Format probability
    if '부족 확률' in df.columns:
        df['부족 확률'] = (df['부족 확률'] * 100).round(1).astype(str) + '%'
    
    return df

def create_time_category_cards(empty_stations_df: pd.DataFrame) -> None:
    """Create time category cards for empty stations"""
    
    categories = {
        "1시간 미만": {"count": 0, "color": "#FFF59D"},
        "1-2시간": {"count": 0, "color": "#FFB74D"},
        "2-3시간": {"count": 0, "color": "#FF7043"},
        "3시간 이상": {"count": 0, "color": "#EF5350"}
    }
    
    # Calculate categories based on stockout probability
    if 'stockout_probability' in empty_stations_df.columns:
        for _, row in empty_stations_df.iterrows():
            if row.get('is_stockout', 0) == 1:
                prob = row.get('stockout_probability', 0)
                if prob < 0.3:
                    categories["1시간 미만"]["count"] += 1
                elif prob < 0.5:
                    categories["1-2시간"]["count"] += 1
                elif prob < 0.7:
                    categories["2-3시간"]["count"] += 1
                else:
                    categories["3시간 이상"]["count"] += 1
    
    # Display cards
    cols = st.columns(4)
    for idx, (label, info) in enumerate(categories.items()):
        with cols[idx]:
            text_color = "white" if idx > 1 else "black"
            st.markdown(f"""
            <div style='background-color: {info["color"]}; 
                        padding: 20px; 
                        border-radius: 10px; 
                        text-align: center;
                        color: {text_color};'>
                <h4 style='margin: 0;'>{label}</h4>
                <h2 style='margin: 10px 0;'>{info["count"]}</h2>
            </div>
            """, unsafe_allow_html=True)

def create_gauge_chart(value: float, title: str, max_value: float = 100) -> go.Figure:
    """Create a gauge chart"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "#5B47FB"},
            'steps': [
                {'range': [0, max_value*0.3], 'color': "#E8F5E9"},
                {'range': [max_value*0.3, max_value*0.7], 'color': "#FFF9C4"},
                {'range': [max_value*0.7, max_value], 'color': "#FFEBEE"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_heatmap(data: pd.DataFrame, x_col: str, y_col: str, 
                   value_col: str, title: str) -> go.Figure:
    """Create a heatmap visualization"""
    
    pivot_data = data.pivot(index=y_col, columns=x_col, values=value_col)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlBu_r',
        reversescale=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=400
    )
    
    return fig

def create_district_map(station_data: pd.DataFrame) -> go.Figure:
    """Create a map visualization of stations by district"""
    
    # Sample coordinates for Seoul districts (실제 좌표로 교체 필요)
    district_coords = {
        "강남구": (37.5172, 127.0473),
        "강동구": (37.5301, 127.1238),
        "강북구": (37.6396, 127.0257),
        "강서구": (37.5509, 126.8495),
        "관악구": (37.4784, 126.9516),
        # Add more districts...
    }
    
    fig = go.Figure()
    
    for district, (lat, lon) in district_coords.items():
        district_stations = station_data[station_data.get('district', '') == district]
        empty_count = len(district_stations[district_stations.get('is_stockout', 0) == 1])
        
        color = 'red' if empty_count > 5 else 'orange' if empty_count > 2 else 'green'
        
        fig.add_trace(go.Scattermapbox(
            mode='markers+text',
            lon=[lon],
            lat=[lat],
            marker={'size': 15 + empty_count * 2, 'color': color},
            text=[f"{district}<br>빈 대여소: {empty_count}"],
            textposition="top center",
            name=district
        ))
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=37.5665, lon=126.9780), zoom=10),
        showlegend=False,
        height=500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    return fig

def create_alert_box(message: str, alert_type: str = "info") -> None:
    """Create an alert box with appropriate styling"""
    
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    colors = {
        "info": "#E3F2FD",
        "success": "#E8F5E9",
        "warning": "#FFF3E0",
        "error": "#FFEBEE"
    }
    
    st.markdown(f"""
    <div style='background-color: {colors.get(alert_type, colors["info"])}; 
                padding: 15px; 
                border-radius: 10px; 
                margin: 10px 0;'>
        <p style='margin: 0; font-size: 16px;'>
            {icons.get(alert_type, icons["info"])} {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_progress_bar(value: float, max_value: float = 100, 
                        label: str = "") -> None:
    """Create a custom progress bar"""
    
    percentage = min(100, (value / max_value) * 100)
    
    color = "#4CAF50" if percentage < 60 else "#FFC107" if percentage < 80 else "#F44336"
    
    st.markdown(f"""
    <div style='margin: 10px 0;'>
        <p style='margin-bottom: 5px;'>{label}</p>
        <div style='background-color: #E0E0E0; 
                    border-radius: 10px; 
                    overflow: hidden;'>
            <div style='background-color: {color}; 
                        width: {percentage}%; 
                        height: 20px; 
                        text-align: center; 
                        color: white;'>
                {value:.0f}/{max_value:.0f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def format_number(num: float, decimals: int = 0) -> str:
    """Format number with Korean style thousands separator"""
    if pd.isna(num):
        return "-"
    
    if decimals > 0:
        formatted = f"{num:,.{decimals}f}"
    else:
        formatted = f"{int(num):,}"
    
    # Convert to Korean style if needed
    return formatted.replace(",", ",")