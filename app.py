from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import math
import copy

app = Flask(__name__)
CORS(app)  # 프론트엔드에서 API 호출 허용

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Haversine formula를 사용하여 두 지점 간의 거리(km)를 계산
    """
    # 지구 반지름 (km)
    R = 6371.0
    
    # 위도와 경도를 라디안으로 변환
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # 위도와 경도의 차이
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def nearest_neighbor_algorithm(stations_data):
    """
    최근접 이웃 알고리즘을 사용하여 최적 경로 계산
    차고지 -> 모든 대여소 방문 -> 차고지 복귀
    """
    # 데이터 준비
    depot = stations_data[stations_data['type'] == 'depot'].iloc[0]
    stations = stations_data[stations_data['type'] == 'station'].copy()
    
    # 경로와 총 거리 초기화
    route = [depot]  # 차고지에서 시작
    total_distance = 0
    unvisited_stations = stations.copy()
    current_location = depot
    
    # 모든 대여소를 방문할 때까지 반복
    while not unvisited_stations.empty:
        # 현재 위치에서 가장 가까운 미방문 대여소 찾기
        min_distance = float('inf')
        nearest_station = None
        nearest_index = None
        
        for idx, station in unvisited_stations.iterrows():
            distance = haversine_distance(
                current_location['lat'], current_location['lon'],
                station['lat'], station['lon']
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_station = station
                nearest_index = idx
        
        # 가장 가까운 대여소로 이동
        route.append(nearest_station)
        total_distance += min_distance
        current_location = nearest_station
        
        # 방문한 대여소를 제거
        unvisited_stations = unvisited_stations.drop(nearest_index)
    
    # 마지막에 차고지로 복귀
    final_distance = haversine_distance(
        current_location['lat'], current_location['lon'],
        depot['lat'], depot['lon']
    )
    route.append(depot)
    total_distance += final_distance
    
    return route, total_distance

@app.route('/')
def index():
    """
    메인 페이지 (index.html) 서빙
    브라우저에서 http://localhost:8080 접속 시 index.html 파일을 반환
    """
    return send_from_directory('.', 'index.html')

@app.route('/api/get-route', methods=['GET'])
def get_optimal_route():
    """
    최적 경로 계산 API 엔드포인트
    """
    try:
        # CSV 파일에서 데이터 로드
        stations_data = pd.read_csv('/Users/idong-ug/Documents/직장/프로젝트/bicycle_analize2/강동구_송파구_공공자전거_데이터.csv')
        
        # 최근접 이웃 알고리즘으로 최적 경로 계산
        route, total_distance = nearest_neighbor_algorithm(stations_data)
        
        # 응답 데이터 구성
        route_data = []
        for station in route:
            route_data.append({
                'id': int(station['id']),
                'name': station['name'],
                'lat': float(station['lat']),
                'lon': float(station['lon']),
                'type': station['type']
            })
        
        response = {
            'success': True,
            'route': route_data,
            'total_distance': round(total_distance, 2),
            'total_stations': len(route) - 2  # 시작과 끝의 차고지 제외
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stations', methods=['GET'])
def get_stations():
    """
    모든 대여소 정보를 반환하는 API
    """
    try:
        stations_data = pd.read_csv('stations.csv')
        stations_list = []
        
        for _, station in stations_data.iterrows():
            stations_list.append({
                'id': int(station['id']),
                'name': station['name'],
                'lat': float(station['lat']),
                'lon': float(station['lon']),
                'type': station['type']
            })
        
        return jsonify({
            'success': True,
            'stations': stations_list
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("따릉이 재배치 경로 최적화 서버 시작...")
    print("API 엔드포인트:")
    print("- GET /api/stations: 모든 대여소 정보")
    print("- GET /api/get-route: 최적 경로 계산")
    app.run(debug=True, host='0.0.0.0', port=8080)