import requests
import json
import math
import time
from collections import defaultdict
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. GeoJSON 기반 구 분류기 (기존 코드 유지)
# ---------------------------------------------------------------------------
class SeoulDistrictClassifier:
    """서울시 구 분류기 (GeoJSON 활용)"""
    def __init__(self):
        self.district_polygons = {}
        self.load_geojson()
    
    def load_geojson(self):
        """GeoJSON 데이터 로드 (온라인 또는 로컬)"""
        self.district_bounds = {
            '강남구': {'min_lat': 37.4687, 'max_lat': 37.5687, 'min_lon': 127.0164, 'max_lon': 127.0964},
            '강동구': {'min_lat': 37.5201, 'max_lat': 37.5501, 'min_lon': 127.1138, 'max_lon': 127.1438},
            '강북구': {'min_lat': 37.6296, 'max_lat': 37.6596, 'min_lon': 127.0157, 'max_lon': 127.0457},
            '강서구': {'min_lat': 37.5409, 'max_lat': 37.5709, 'min_lon': 126.8295, 'max_lon': 126.8795},
            '관악구': {'min_lat': 37.4684, 'max_lat': 37.4984, 'min_lon': 126.9416, 'max_lon': 126.9716},
            '광진구': {'min_lat': 37.5284, 'max_lat': 37.5584, 'min_lon': 127.0722, 'max_lon': 127.1022},
            '구로구': {'min_lat': 37.4854, 'max_lat': 37.5154, 'min_lon': 126.8774, 'max_lon': 126.9074},
            '금천구': {'min_lat': 37.4467, 'max_lat': 37.4767, 'min_lon': 126.8854, 'max_lon': 126.9154},
            '노원구': {'min_lat': 37.6442, 'max_lat': 37.6742, 'min_lon': 127.0468, 'max_lon': 127.0768},
            '도봉구': {'min_lat': 37.6587, 'max_lat': 37.6887, 'min_lon': 127.0371, 'max_lon': 127.0671},
            '동대문구': {'min_lat': 37.5644, 'max_lat': 37.5944, 'min_lon': 127.0299, 'max_lon': 127.0599},
            '동작구': {'min_lat': 37.5024, 'max_lat': 37.5324, 'min_lon': 126.9293, 'max_lon': 126.9593},
            '마포구': {'min_lat': 37.5537, 'max_lat': 37.5837, 'min_lon': 126.8987, 'max_lon': 126.9287},
            '서대문구': {'min_lat': 37.5691, 'max_lat': 37.5991, 'min_lon': 126.9268, 'max_lon': 126.9568},
            '서초구': {'min_lat': 37.4737, 'max_lat': 37.5037, 'min_lon': 127.0225, 'max_lon': 127.0525},
            '성동구': {'min_lat': 37.5533, 'max_lat': 37.5833, 'min_lon': 127.0269, 'max_lon': 127.0569},
            '성북구': {'min_lat': 37.5794, 'max_lat': 37.6094, 'min_lon': 127.0067, 'max_lon': 127.0367},
            '송파구': {'min_lat': 37.5046, 'max_lat': 37.5346, 'min_lon': 127.0950, 'max_lon': 127.1250},
            '양천구': {'min_lat': 37.5067, 'max_lat': 37.5367, 'min_lon': 126.8565, 'max_lon': 126.8865},
            '영등포구': {'min_lat': 37.5164, 'max_lat': 37.5464, 'min_lon': 126.8863, 'max_lon': 126.9163},
            '용산구': {'min_lat': 37.5284, 'max_lat': 37.5584, 'min_lon': 126.9554, 'max_lon': 126.9854},
            '은평구': {'min_lat': 37.6076, 'max_lat': 37.6376, 'min_lon': 126.9127, 'max_lon': 126.9427},
            '종로구': {'min_lat': 37.5635, 'max_lat': 37.5935, 'min_lon': 126.9691, 'max_lon': 126.9991},
            '중구': {'min_lat': 37.5540, 'max_lat': 37.5840, 'min_lon': 126.9879, 'max_lon': 127.0179},
            '중랑구': {'min_lat': 37.5963, 'max_lat': 37.6263, 'min_lon': 127.0825, 'max_lon': 127.1125}
        }
        
        self.district_centers = {
            '강남구': (37.5172, 127.0473), '강동구': (37.5301, 127.1238),
            '강북구': (37.6396, 127.0257), '강서구': (37.5509, 126.8495),
            '관악구': (37.4784, 126.9516), '광진구': (37.5384, 127.0822),
            '구로구': (37.4954, 126.8874), '금천구': (37.4567, 126.8954),
            '노원구': (37.6542, 127.0568), '도봉구': (37.6687, 127.0471),
            '동대문구': (37.5744, 127.0399), '동작구': (37.5124, 126.9393),
            '마포구': (37.5637, 126.9087), '서대문구': (37.5791, 126.9368),
            '서초구': (37.4837, 127.0325), '성동구': (37.5633, 127.0369),
            '성북구': (37.5894, 127.0167), '송파구': (37.5146, 127.1050),
            '양천구': (37.5167, 126.8665), '영등포구': (37.5264, 126.8963),
            '용산구': (37.5384, 126.9654), '은평구': (37.6176, 126.9227),
            '종로구': (37.5735, 126.9791), '중구': (37.5640, 126.9979),
            '중랑구': (37.6063, 127.0925)
        }
    
    def find_district(self, lat, lon):
        """좌표가 속한 구를 찾습니다"""
        for district, bounds in self.district_bounds.items():
            if (bounds['min_lat'] <= lat <= bounds['max_lat'] and 
                bounds['min_lon'] <= lon <= bounds['max_lon']):
                return district
        
        min_distance = float('inf')
        closest_district = None
        
        for district, (center_lat, center_lon) in self.district_centers.items():
            distance = math.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            if distance < min_distance:
                min_distance = distance
                closest_district = district
        
        return closest_district

# ---------------------------------------------------------------------------
# 🆕 2. 클러스터링 모듈
# ---------------------------------------------------------------------------
class BikeStationClusterer:
    """따릉이 대여소 클러스터링"""
    
    def __init__(self, num_vehicles=3):
        self.num_vehicles = num_vehicles
        self.clusters = {}
        
    def create_balanced_clusters(self, stations):
        """작업량 균형을 고려한 클러스터 생성"""
        
        # 수거/배송 필요 대여소 분리
        pickup_stations = []
        delivery_stations = []
        
        for station in stations:
            if station.get('pickup', 0) > 0:
                pickup_stations.append(station)
            elif station.get('delivery', 0) > 0:
                delivery_stations.append(station)
        
        print(f"\n📊 클러스터링 시작:")
        print(f"  - 수거 필요: {len(pickup_stations)}개")
        print(f"  - 배송 필요: {len(delivery_stations)}개")
        print(f"  - 트럭 수: {self.num_vehicles}대")
        
        # 1차 클러스터링: 수거/배송 별도
        pickup_clusters = self._kmeans_clustering(
            pickup_stations, 
            n_clusters=min(self.num_vehicles, len(pickup_stations))
        ) if pickup_stations else {}
        
        delivery_clusters = self._kmeans_clustering(
            delivery_stations,
            n_clusters=min(self.num_vehicles, len(delivery_stations))
        ) if delivery_stations else {}
        
        # 2차 클러스터링: 인접 클러스터 병합
        final_clusters = self._merge_clusters(pickup_clusters, delivery_clusters)
        
        return final_clusters
    
    def _kmeans_clustering(self, stations, n_clusters):
        """K-means 클러스터링 수행"""
        if not stations or n_clusters == 0:
            return {}
            
        # 좌표와 작업량을 특징으로 사용
        features = []
        for station in stations:
            features.append([
                station['lat'],
                station['lon'],
                station.get('pickup', 0) + station.get('delivery', 0)  # 작업량
            ])
        
        # 정규화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # 클러스터별로 스테이션 그룹화
        clusters = defaultdict(list)
        for station, label in zip(stations, labels):
            clusters[label].append(station)
        
        return dict(clusters)
    
    def _merge_clusters(self, pickup_clusters, delivery_clusters):
        """수거/배송 클러스터를 지리적 근접성 기반으로 병합"""
        final_clusters = []
        
        # 각 클러스터의 중심점 계산
        pickup_centers = {}
        for idx, stations in pickup_clusters.items():
            if stations:
                avg_lat = np.mean([s['lat'] for s in stations])
                avg_lon = np.mean([s['lon'] for s in stations])
                pickup_centers[f'pickup_{idx}'] = {
                    'center': (avg_lat, avg_lon),
                    'stations': stations
                }
        
        delivery_centers = {}
        for idx, stations in delivery_clusters.items():
            if stations:
                avg_lat = np.mean([s['lat'] for s in stations])
                avg_lon = np.mean([s['lon'] for s in stations])
                delivery_centers[f'delivery_{idx}'] = {
                    'center': (avg_lat, avg_lon),
                    'stations': stations
                }
        
        # 가장 가까운 수거/배송 클러스터 페어링
        used_delivery = set()
        
        for pickup_key, pickup_data in pickup_centers.items():
            cluster_stations = pickup_data['stations'].copy()
            
            # 가장 가까운 배송 클러스터 찾기
            min_dist = float('inf')
            closest_delivery = None
            
            for delivery_key, delivery_data in delivery_centers.items():
                if delivery_key not in used_delivery:
                    dist = self._calculate_distance(
                        pickup_data['center'][0], pickup_data['center'][1],
                        delivery_data['center'][0], delivery_data['center'][1]
                    )
                    if dist < min_dist:
                        min_dist = dist
                        closest_delivery = delivery_key
            
            if closest_delivery:
                cluster_stations.extend(delivery_centers[closest_delivery]['stations'])
                used_delivery.add(closest_delivery)
            
            final_clusters.append(cluster_stations)
        
        # 페어링되지 않은 배송 클러스터 추가
        for delivery_key, delivery_data in delivery_centers.items():
            if delivery_key not in used_delivery:
                final_clusters.append(delivery_data['stations'])
        
        # 빈 클러스터가 있으면 트럭 수에 맞게 조정
        while len(final_clusters) < self.num_vehicles:
            final_clusters.append([])
        
        print(f"\n✅ 클러스터링 완료: {len(final_clusters)}개 클러스터 생성")
        for i, cluster in enumerate(final_clusters):
            if cluster:
                pickup_count = sum(1 for s in cluster if s.get('pickup', 0) > 0)
                delivery_count = sum(1 for s in cluster if s.get('delivery', 0) > 0)
                print(f"  클러스터 {i+1}: 총 {len(cluster)}개 (수거 {pickup_count}, 배송 {delivery_count})")
        
        return final_clusters
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """두 지점 간 거리 계산 (km)"""
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

# ---------------------------------------------------------------------------
# 3. 데이터 수집 및 구별 분류 (기존 코드 유지)
# ---------------------------------------------------------------------------
def get_bike_station_data_by_district(api_key):
    """따릉이 데이터를 수집하고 구별로 분류합니다"""
    
    print("="*70)
    print("STEP 1: 데이터 수집 및 구별 분류")
    print("="*70)
    
    all_stations = []
    for start_index in [1, 1001, 2001]:
        end_index = start_index + 999
        url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/bikeList/{start_index}/{end_index}/"
        try:
            response = requests.get(url)
            data = response.json()
            if 'rentBikeStatus' in data and 'row' in data['rentBikeStatus']:
                all_stations.extend(data['rentBikeStatus']['row'])
        except:
            continue
    
    print(f"✓ 총 {len(all_stations)}개 대여소 데이터 수집 완료")
    
    classifier = SeoulDistrictClassifier()
    district_stations = defaultdict(list)
    
    for station in all_stations:
        try:
            lat = float(station['stationLatitude'])
            lon = float(station['stationLongitude'])
            district = classifier.find_district(lat, lon)
            
            if district:
                station['district'] = district
                district_stations[district].append(station)
        except:
            continue
    
    print(f"✓ {len(district_stations)}개 구로 분류 완료")
    
    for district, stations in sorted(district_stations.items()):
        print(f"  - {district}: {len(stations)}개 대여소")
    
    return district_stations

# ---------------------------------------------------------------------------
# 4. 구별 재배치 분석 (기존 코드 유지)
# ---------------------------------------------------------------------------
def analyze_district_redistribution_needs(district_stations):
    """구별 재배치 필요도를 분석합니다"""
    
    print("\n" + "="*70)
    print("STEP 2: 구별 재배치 필요도 분석")
    print("="*70)
    
    district_analysis = {}
    
    for district, stations in district_stations.items():
        pickup_needed = []
        delivery_needed = []
        total_imbalance = 0
        
        for station in stations:
            try:
                racks = int(station['rackTotCnt'])
                bikes = int(station['parkingBikeTotCnt'])
                
                if racks == 0:
                    continue
                
                occupancy = int(station['shared'])
                target = racks * 0.7
                imbalance = bikes - target
                
                station_info = {
                    'id': station['stationId'],
                    'name': station['stationName'],
                    'lat': float(station['stationLatitude']),
                    'lon': float(station['stationLongitude']),
                    'bikes': bikes,
                    'racks': racks,
                    'occupancy': occupancy,
                    'imbalance': abs(imbalance)
                }
                
                if occupancy >= 130:
                    station_info['pickup'] = min(int(imbalance), 10)
                    station_info['delivery'] = 0
                    pickup_needed.append(station_info)
                    total_imbalance += station_info['pickup']
                elif occupancy <= 30:
                    station_info['pickup'] = 0
                    station_info['delivery'] = min(int(abs(imbalance)), 10)
                    delivery_needed.append(station_info)
                    total_imbalance += station_info['delivery']
                    
            except:
                continue
        
        district_analysis[district] = {
            'total_stations': len(stations),
            'pickup_needed': pickup_needed,
            'delivery_needed': delivery_needed,
            'total_imbalance': total_imbalance,
            'urgency_score': len(pickup_needed) + len(delivery_needed)
        }
    
    sorted_districts = sorted(district_analysis.items(), 
                            key=lambda x: x[1]['urgency_score'], 
                            reverse=True)
    
    print(f"\n{'구':<10} {'대여소':<8} {'수거필요':<10} {'배송필요':<10} {'긴급도':<8}")
    print("-"*50)
    
    for district, analysis in sorted_districts[:10]:
        print(f"{district:<10} {analysis['total_stations']:<8} "
              f"{len(analysis['pickup_needed']):<10} "
              f"{len(analysis['delivery_needed']):<10} "
              f"{analysis['urgency_score']:<8}")
    
    return district_analysis

# ---------------------------------------------------------------------------
# 🆕 5. 클러스터 기반 OR-Tools 최적화
# ---------------------------------------------------------------------------
def solve_district_with_clustering(district_name, analysis, num_vehicles=2, vehicle_capacity=20):
    """클러스터링 기반 구별 재배치 최적화"""
    
    problem_stations = analysis['pickup_needed'] + analysis['delivery_needed']
    
    if len(problem_stations) == 0:
        return None
    
    print(f"\n{'='*70}")
    print(f"🚀 {district_name} 클러스터링 기반 최적화")
    print(f"{'='*70}")
    print(f"📌 문제 크기: {len(problem_stations)}개 대여소")
    
    # 노드 수가 적으면 클러스터링 없이 직접 처리
    if len(problem_stations) <= 30:
        print("  → 소규모 문제: 클러스터링 없이 직접 최적화")
        return solve_single_cluster_with_ortools(
            district_name, problem_stations, num_vehicles, vehicle_capacity
        )
    
    # 클러스터링 수행
    clusterer = BikeStationClusterer(num_vehicles)
    clusters = clusterer.create_balanced_clusters(problem_stations)
    
    # 각 클러스터별로 OR-Tools 적용
    all_routes = []
    total_distance = 0
    solution_methods = []  # 각 클러스터의 해결 방법 기록
    
    for i, cluster_stations in enumerate(clusters):
        if not cluster_stations:
            continue
        
        print(f"\n📦 클러스터 {i+1}/{len(clusters)} 처리 중...")
        
        # 단일 트럭으로 클러스터 해결
        solution = solve_single_cluster_with_ortools(
            district_name,  # 구 이름을 전달하여 올바른 차고지 사용
            cluster_stations,
            num_vehicles=1,  # 각 클러스터는 1대의 트럭이 담당
            vehicle_capacity=vehicle_capacity,
            cluster_id=i+1  # 클러스터 ID 전달
        )
        
        if solution and solution['routes']:
            # 트럭 ID 조정
            for route in solution['routes']:
                route['vehicle_id'] = i
                route['cluster_id'] = i+1
            all_routes.extend(solution['routes'])
            total_distance += solution.get('total_distance', 0)
            solution_methods.append(solution.get('method', 'Unknown'))
    
    return {
        'routes': all_routes,
        'total_distance': total_distance,
        'clustering_used': True,
        'num_clusters': len(clusters),
        'solution_methods': solution_methods
    }

def solve_single_cluster_with_ortools(district_name, stations, num_vehicles=1, vehicle_capacity=20, cluster_id=None):
    """단일 클러스터에 대한 OR-Tools 최적화 (개선된 버전)"""
    
    if not stations:
        return None
    
    # 문제 실행 가능성 체크
    total_pickup = sum(s.get('pickup', 0) for s in stations)
    total_delivery = sum(s.get('delivery', 0) for s in stations)
    '''
    # 만약 수거량이나 배송량이 트럭 용량을 크게 초과하면 바로 휴리스틱 사용
    if total_pickup > vehicle_capacity * num_vehicles * 2 or total_delivery > vehicle_capacity * num_vehicles * 2:
        print(f"  ⚠ 문제 규모가 너무 큼 (수거: {total_pickup}, 배송: {total_delivery})")
        classifier = SeoulDistrictClassifier()
        actual_district = district_name.split('_')[0] if '_' in district_name else district_name
        if actual_district in classifier.district_centers:
            depot_lat, depot_lon = classifier.district_centers[actual_district]
        else:
            depot_lat = np.mean([s['lat'] for s in stations])
            depot_lon = np.mean([s['lon'] for s in stations])
        
        depot = {
            'name': f'{actual_district} 차고지',
            'lat': depot_lat,
            'lon': depot_lon
        }
        return solve_with_heuristic(district_name, stations, num_vehicles, vehicle_capacity, depot)
    '''
    # 구별 고정 차고지 사용
    classifier = SeoulDistrictClassifier()
    
    # district_name이 클러스터 이름 형식(예: "강남구_C1")인 경우 실제 구 이름만 추출
    actual_district = district_name.split('_')[0] if '_' in district_name else district_name
    
    # 해당 구의 고정 차고지 좌표 가져오기
    if actual_district in classifier.district_centers:
        depot_lat, depot_lon = classifier.district_centers[actual_district]
    else:
        # 만약 구 정보가 없으면 클러스터 중심점 사용 (fallback)
        depot_lat = np.mean([s['lat'] for s in stations])
        depot_lon = np.mean([s['lon'] for s in stations])
    
    depot = {
        'id': f'DEPOT_{actual_district}',
        'name': f'{actual_district} 차고지',
        'lat': depot_lat,
        'lon': depot_lon,
        'pickup': 0,
        'delivery': 0
    }
    
    nodes = [depot] + stations
    pickups = [0] + [s.get('pickup', 0) for s in stations]
    deliveries = [0] + [s.get('delivery', 0) for s in stations]
    
    # 거리 행렬 계산
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return int(R * c * 1000)
    
    distance_matrix = []
    for from_node in nodes:
        row = []
        for to_node in nodes:
            dist = haversine(from_node['lat'], from_node['lon'],
                           to_node['lat'], to_node['lon'])
            row.append(dist)
        distance_matrix.append(row)
    
    # OR-Tools 모델 생성
    manager = pywrapcp.RoutingIndexManager(len(nodes), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # 거리 콜백
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 용량 제약 - 개선된 버전
    # 초기 적재량을 설정 (수거/배송 균형을 위해)
    initial_load = min(vehicle_capacity // 2, sum(deliveries))  # 초기에 일부 자전거를 적재한 상태로 시작
    
    def capacity_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return pickups[from_node] - deliveries[from_node]
    
    capacity_callback_index = routing.RegisterUnaryTransitCallback(capacity_callback)
    
    # 용량 제약 추가 시 여유를 둠
    routing.AddDimensionWithVehicleCapacity(
        capacity_callback_index,
        0,  # slack
        [vehicle_capacity] * num_vehicles,  # 각 차량의 최대 용량
        False,  # start_cumul_to_zero를 False로 설정하여 초기 적재 허용
        "Capacity"
    )
    
    # 각 차량의 초기 적재량 설정
    capacity_dimension = routing.GetDimensionOrDie("Capacity")
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        capacity_dimension.SetCumulVarSoftLowerBound(index, initial_load, 1000)
        capacity_dimension.SetCumulVarSoftUpperBound(index, initial_load, 1000)
    
    # 거리 균등화
    routing.AddDimension(
        transit_callback_index,
        0,
        50000,
        True,
        "Distance"
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    # 탐색 파라미터 - 단순하고 빠른 설정
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # 가장 기본적인 전략으로 시작
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    # 시간 제한 설정
    search_parameters.time_limit.FromSeconds(30)  # 10초로 제한
    
    # 문제 해결
    print(f"  🔄 OR-Tools 시도 중... (최대 30초)")
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        result = extract_solution(manager, routing, solution, nodes, pickups, deliveries, num_vehicles)
        result['method'] = 'OR-Tools'
        print(f"  ✅ OR-Tools로 경로 생성 성공")
        return result
    else:
        print(f"  ⚠ OR-Tools 실패, 휴리스틱 사용")
        result = solve_with_heuristic(district_name, stations, num_vehicles, vehicle_capacity, depot)
        result['method'] = 'Heuristic'
        return result

# ---------------------------------------------------------------------------
# 6. 솔루션 추출 및 포맷팅 (기존 코드 유지)
# ---------------------------------------------------------------------------
def extract_solution(manager, routing, solution, nodes, pickups, deliveries, num_vehicles):
    """OR-Tools 솔루션에서 경로 정보를 추출합니다"""
    
    routes = []
    total_distance = 0
    
    for vehicle_id in range(num_vehicles):
        route = {
            'vehicle_id': vehicle_id,
            'path': [],
            'distance': 0,
            'pickups': 0,
            'deliveries': 0,
            'load_changes': []
        }
        
        index = routing.Start(vehicle_id)
        current_load = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            node = nodes[node_index]
            pickup = pickups[node_index]
            delivery = deliveries[node_index]
            current_load += pickup - delivery
            
            route['path'].append({
                'name': node['name'],
                'pickup': pickup,
                'delivery': delivery,
                'current_load': current_load
            })
            
            route['pickups'] += pickup
            route['deliveries'] += delivery
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route['distance'] += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        
        node_index = manager.IndexToNode(index)
        route['path'].append({
            'name': nodes[node_index]['name'],
            'pickup': 0,
            'delivery': 0,
            'current_load': current_load
        })
        
        if len(route['path']) > 2:
            routes.append(route)
            total_distance += route['distance']
    
    return {
        'routes': routes,
        'total_distance': total_distance
    }

# ---------------------------------------------------------------------------
# 7. 휴리스틱 솔버 (개선 - 거리 계산 추가)
# ---------------------------------------------------------------------------
def solve_with_heuristic(district_name, stations, num_vehicles, vehicle_capacity, depot=None):
    """대규모 문제를 위한 휴리스틱 해법 (거리 계산 포함)"""
    
    # depot이 없으면 구의 중심점 사용
    if depot is None:
        classifier = SeoulDistrictClassifier()
        actual_district = district_name.split('_')[0] if '_' in district_name else district_name
        if actual_district in classifier.district_centers:
            depot_lat, depot_lon = classifier.district_centers[actual_district]
        else:
            depot_lat = np.mean([s['lat'] for s in stations]) if stations else 37.5665
            depot_lon = np.mean([s['lon'] for s in stations]) if stations else 126.9780
        
        depot = {
            'name': f'{actual_district} 차고지',
            'lat': depot_lat,
            'lon': depot_lon
        }
    
    # Haversine 거리 계산 함수
    def calculate_distance(lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c * 1000  # 미터 단위
    
    pickup_stations = [s for s in stations if s.get('pickup', 0) > 0]
    delivery_stations = [s for s in stations if s.get('delivery', 0) > 0]
    
    pickup_stations.sort(key=lambda x: x.get('pickup', 0), reverse=True)
    delivery_stations.sort(key=lambda x: x.get('delivery', 0), reverse=True)
    
    routes = []
    total_distance = 0
    
    for vehicle_id in range(num_vehicles):
        route = {
            'vehicle_id': vehicle_id,
            'path': [{'name': depot['name'], 'pickup': 0, 'delivery': 0, 'current_load': 0}],
            'distance': 0,
            'pickups': 0,
            'deliveries': 0
        }
        
        current_load = 0
        current_lat, current_lon = depot['lat'], depot['lon']
        
        # 수거 작업
        for station in pickup_stations[vehicle_id::num_vehicles]:
            if current_load + station['pickup'] <= vehicle_capacity:
                # 거리 계산 및 누적
                dist = calculate_distance(current_lat, current_lon, station['lat'], station['lon'])
                route['distance'] += dist
                
                current_load += station['pickup']
                current_lat, current_lon = station['lat'], station['lon']
                
                route['path'].append({
                    'name': station['name'],
                    'pickup': station['pickup'],
                    'delivery': 0,
                    'current_load': current_load
                })
                route['pickups'] += station['pickup']
        
        # 배송 작업
        for station in delivery_stations[vehicle_id::num_vehicles]:
            if current_load >= station['delivery']:
                # 거리 계산 및 누적
                dist = calculate_distance(current_lat, current_lon, station['lat'], station['lon'])
                route['distance'] += dist
                
                current_load -= station['delivery']
                current_lat, current_lon = station['lat'], station['lon']
                
                route['path'].append({
                    'name': station['name'],
                    'pickup': 0,
                    'delivery': station['delivery'],
                    'current_load': current_load
                })
                route['deliveries'] += station['delivery']
        
        # 차고지 복귀 거리 추가
        if len(route['path']) > 1:  # 실제 작업이 있었다면
            dist = calculate_distance(current_lat, current_lon, depot['lat'], depot['lon'])
            route['distance'] += dist
        
        route['path'].append({
            'name': depot['name'],
            'pickup': 0,
            'delivery': 0,
            'current_load': current_load
        })
        
        if len(route['path']) > 2:
            routes.append(route)
            total_distance += route['distance']
    
    print(f"  ✅ 휴리스틱으로 경로 생성 완료")
    
    return {
        'routes': routes, 
        'total_distance': total_distance,
        'method': 'Heuristic'
    }

# ---------------------------------------------------------------------------
# 8. 결과 출력 (개선)
# ---------------------------------------------------------------------------
def print_district_solution(district_name, solution):
    """구별 솔루션을 보기 좋게 출력합니다"""
    
    print(f"\n{'='*70}")
    print(f"📍 {district_name} 재배치 계획")
    
    # 해결 방법 표시
    if solution:
        if solution.get('clustering_used'):
            print(f"🔧 클러스터링 사용: {solution.get('num_clusters', 0)}개 클러스터")
            if 'solution_methods' in solution:
                methods = solution.get('solution_methods', [])
                or_tools_count = methods.count('OR-Tools')
                heuristic_count = methods.count('Heuristic')
                print(f"📊 해결 방법: OR-Tools {or_tools_count}개, Heuristic {heuristic_count}개")
        else:
            method = solution.get('method', 'Unknown')
            print(f"🔧 해결 방법: {method}")
    
    print(f"{'='*70}")
    
    if not solution or not solution['routes']:
        print("재배치가 필요없거나 경로 생성 실패")
        return
    
    # 총 거리 표시
    if solution.get('total_distance', 0) > 0:
        print(f"\n📏 총 이동 거리: {solution['total_distance']/1000:.2f}km")
    
    for route in solution['routes']:
        cluster_info = f" (클러스터 {route.get('cluster_id')})" if 'cluster_id' in route else ""
        print(f"\n🚚 트럭 {route['vehicle_id'] + 1}번{cluster_info}")
        
        if route['distance'] > 0:
            print(f"이동 거리: {route['distance']/1000:.2f}km")
        
        print(f"수거: {route['pickups']}대, 배송: {route['deliveries']}대")
        print("\n경로:")
        
        for i, stop in enumerate(route['path']):
            if stop['pickup'] > 0:
                print(f"  {i}. ↗️ {stop['name']}: +{stop['pickup']}대 (적재: {stop['current_load']})")
            elif stop['delivery'] > 0:
                print(f"  {i}. ↘️ {stop['name']}: -{stop['delivery']}대 (적재: {stop['current_load']})")
            else:
                print(f"  {i}. 📍 {stop['name']}")

# ---------------------------------------------------------------------------
# 9. 메인 실행 함수 (수정)
# ---------------------------------------------------------------------------
def main():
    """통합 재배치 시스템 메인 함수"""
    
    print("\n" + "="*70)
    print(" "*20 + "서울시 따릉이 통합 재배치 시스템")
    print(" "*20 + "🆕 클러스터링 기반 최적화")
    print(" "*25 + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    API_KEY = "6464716442737069363863566b466c"
    
    # 1. 데이터 수집 및 구별 분류
    district_stations = get_bike_station_data_by_district(API_KEY)
    
    # 2. 구별 재배치 필요도 분석
    district_analysis = analyze_district_redistribution_needs(district_stations)
    
    # 3. 우선순위 높은 구 선택
    sorted_districts = sorted(district_analysis.items(),
                            key=lambda x: x[1]['urgency_score'],
                            reverse=True)
    
    print("\n" + "="*70)
    print("STEP 3: 재배치 계획 수립")
    print("="*70)
    
    print("\n처리 방식을 선택하세요:")
    print("1. 특정 구 선택 처리 (클러스터링)")
    print("2. 특정 구 선택 처리 (기존 방식)")
    print("3. 구별 상세 분석만 보기")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    if choice in ['1', '2']:
        # 특정 구 선택
        print("\n사용 가능한 구:")
        for i, (district, _) in enumerate(sorted_districts[:25], 1):
            print(f"{i}. {district}")
        
        try:
            idx = int(input("\n처리할 구 번호 선택: ")) - 1
            if 0 <= idx < len(sorted_districts):
                district, analysis = sorted_districts[idx]
                
                num_vehicles = int(input(f"투입할 트럭 수 (권장: {max(1, analysis['urgency_score']//10)}): ") or 2)
                capacity = int(input("트럭 용량 (기본: 20): ") or 20)
                
                if choice == '1':
                    # 🆕 클러스터링 기반 해결
                    solution = solve_district_with_clustering(
                        district, analysis, num_vehicles, capacity
                    )
                else:
                    # 기존 방식
                    solution = solve_single_cluster_with_ortools(
                        district, 
                        analysis['pickup_needed'] + analysis['delivery_needed'],
                        num_vehicles, 
                        capacity
                    )
                
                if solution:
                    print_district_solution(district, solution)
                    
                    save = input("\n결과를 JSON으로 저장하시겠습니까? (y/n): ")
                    if save.lower() == 'y':
                        filename = f"{district}_redistribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(solution, f, ensure_ascii=False, indent=2)
                        print(f"✓ {filename}에 저장되었습니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
    
    elif choice == '3':
        # 구별 상세 분석
        for district, analysis in sorted_districts[:25]:
            print(f"\n{district}:")
            print(f"  - 전체 대여소: {analysis['total_stations']}개")
            print(f"  - 수거 필요: {len(analysis['pickup_needed'])}개")
            print(f"  - 배송 필요: {len(analysis['delivery_needed'])}개")
            print(f"  - 총 불균형: {analysis['total_imbalance']}대")
    
    print("\n" + "="*70)
    print("프로그램 종료")
    print("="*70)

if __name__ == "__main__":
    main()