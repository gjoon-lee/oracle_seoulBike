"""
Fixed EDA Analysis - Correcting the issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from db_connection import BikeDataDB

def analyze_station_patterns_correctly(db):
    """Fix the station pattern analysis"""
    print("\n🔧 FIXED Station Pattern Analysis")
    print("="*50)
    
    # Get hourly patterns for each station
    station_patterns = db.read_query("""
        WITH hourly_stats AS (
            SELECT 
                station_id,
                flow_hour,
                SUM(bikes_departed) as total_departures,
                SUM(bikes_arrived) as total_arrivals,
                AVG(bikes_departed) as avg_departures,
                AVG(bikes_arrived) as avg_arrivals
            FROM station_hourly_flow
            GROUP BY station_id, flow_hour
        ),
        station_totals AS (
            SELECT 
                station_id,
                SUM(bikes_departed) as total_station_departures,
                SUM(bikes_arrived) as total_station_arrivals,
                AVG(bikes_departed + bikes_arrived) as avg_activity
            FROM station_hourly_flow
            GROUP BY station_id
        )
        SELECT 
            h.station_id,
            h.flow_hour,
            h.avg_departures,
            h.avg_arrivals,
            t.avg_activity,
            CASE 
                WHEN t.total_station_departures > 0 
                THEN h.total_departures::FLOAT / t.total_station_departures 
                ELSE 0 
            END as hourly_departure_ratio
        FROM hourly_stats h
        JOIN station_totals t ON h.station_id = t.station_id
        WHERE t.avg_activity > 5  -- Filter low-activity stations
        ORDER BY h.station_id, h.flow_hour
    """)
    
    # Calculate proper ratios
    morning_ratios = station_patterns[station_patterns['flow_hour'].between(7, 9)].groupby('station_id')['hourly_departure_ratio'].sum()
    evening_ratios = station_patterns[station_patterns['flow_hour'].between(18, 20)].groupby('station_id')['hourly_departure_ratio'].sum()
    
    print(f"Stations with >20% morning departures: {(morning_ratios > 0.2).sum()}")
    print(f"Stations with >20% evening departures: {(evening_ratios > 0.2).sum()}")
    
    # Show hourly pattern for a busy station
    busy_station = station_patterns[station_patterns['station_id'] == 'ST-99']
    
    plt.figure(figsize=(12, 6))
    plt.plot(busy_station['flow_hour'], busy_station['avg_departures'], 'r-', label='Departures', marker='o')
    plt.plot(busy_station['flow_hour'], busy_station['avg_arrivals'], 'g-', label='Arrivals', marker='o')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Bikes')
    plt.title('Station ST-99 Daily Pattern')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))
    plt.show()
    
    return station_patterns

def analyze_net_flow_patterns(db):
    """Analyze net flow patterns properly"""
    print("\n📊 Net Flow Analysis (Arrivals - Departures)")
    print("="*50)
    
    net_flow_hourly = db.read_query("""
        SELECT 
            flow_hour,
            SUM(bikes_arrived - bikes_departed) as total_net_flow,
            AVG(bikes_arrived - bikes_departed) as avg_net_flow,
            STDDEV(bikes_arrived - bikes_departed) as std_net_flow,
            COUNT(DISTINCT station_id) as stations_active
        FROM station_hourly_flow
        GROUP BY flow_hour
        ORDER BY flow_hour
    """)
    
    print("Hourly Net Flow Pattern:")
    print(net_flow_hourly[['flow_hour', 'avg_net_flow', 'std_net_flow']])
    
    # Plot net flow
    plt.figure(figsize=(12, 6))
    bars = plt.bar(net_flow_hourly['flow_hour'], net_flow_hourly['avg_net_flow'], 
                    color=['red' if x < 0 else 'green' for x in net_flow_hourly['avg_net_flow']])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Net Flow (Arrivals - Departures)')
    plt.title('System-wide Net Flow by Hour')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key hours
    for i, (hour, flow) in enumerate(zip(net_flow_hourly['flow_hour'], net_flow_hourly['avg_net_flow'])):
        if abs(flow) > 0.5:  # Annotate significant flows
            plt.text(hour, flow + 0.05 * np.sign(flow), f'{flow:.2f}', 
                    ha='center', va='bottom' if flow > 0 else 'top', fontsize=8)
    plt.show()
    
def check_data_balance(db):
    """Check arrival vs departure data balance"""
    print("\n⚖️ Data Balance Check")
    print("="*50)
    
    balance_check = db.read_query("""
        SELECT 
            aggregation_type,
            COUNT(*) as record_count,
            SUM(trip_count) as total_trips,
            AVG(trip_count) as avg_trips_per_record
        FROM raw_bike_trips
        GROUP BY aggregation_type
    """)
    
    print(balance_check)
    
    # Check if we have both arrival and departure records
    station_coverage = db.read_query("""
        SELECT 
            COUNT(DISTINCT CASE WHEN aggregation_type = 'departure' THEN start_station_id END) as departure_stations,
            COUNT(DISTINCT CASE WHEN aggregation_type = 'arrival' THEN end_station_id END) as arrival_stations
        FROM raw_bike_trips
    """)
    
    print(f"\nStations with departure data: {station_coverage.iloc[0]['departure_stations']}")
    print(f"Stations with arrival data: {station_coverage.iloc[0]['arrival_stations']}")

def identify_real_station_types(db):
    """Properly identify station types based on flow patterns"""
    print("\n🎯 Correct Station Type Identification")
    print("="*50)
    
    # Get comprehensive station profiles
    profiles = db.read_query("""
        WITH station_hourly_patterns AS (
            SELECT 
                station_id,
                flow_hour,
                AVG(bikes_departed) as avg_dep,
                AVG(bikes_arrived) as avg_arr,
                AVG(bikes_arrived - bikes_departed) as avg_net_flow
            FROM station_hourly_flow
            GROUP BY station_id, flow_hour
        ),
        station_characteristics AS (
            SELECT 
                s.station_id,
                -- Total activity
                AVG(s.bikes_departed + s.bikes_arrived) as avg_total_activity,
                
                -- Morning pattern (7-9 AM)
                AVG(CASE WHEN s.flow_hour BETWEEN 7 AND 9 THEN s.bikes_departed ELSE 0 END) as morning_dep,
                AVG(CASE WHEN s.flow_hour BETWEEN 7 AND 9 THEN s.bikes_arrived ELSE 0 END) as morning_arr,
                
                -- Evening pattern (18-20)
                AVG(CASE WHEN s.flow_hour BETWEEN 18 AND 20 THEN s.bikes_departed ELSE 0 END) as evening_dep,
                AVG(CASE WHEN s.flow_hour BETWEEN 18 AND 20 THEN s.bikes_arrived ELSE 0 END) as evening_arr,
                
                -- Weekend vs weekday
                AVG(CASE WHEN s.is_weekend THEN s.bikes_departed + s.bikes_arrived ELSE 0 END) as weekend_activity,
                AVG(CASE WHEN NOT s.is_weekend THEN s.bikes_departed + s.bikes_arrived ELSE 0 END) as weekday_activity,
                
                -- Variability
                STDDEV(s.bikes_departed + s.bikes_arrived) as activity_std
                
            FROM station_hourly_flow s
            GROUP BY s.station_id
        )
        SELECT 
            station_id,
            avg_total_activity,
            morning_dep,
            morning_arr,
            evening_dep,
            evening_arr,
            weekend_activity,
            weekday_activity,
            CASE 
                WHEN morning_dep > morning_arr * 1.5 AND evening_arr > evening_dep * 1.5 THEN 'Residential'
                WHEN morning_arr > morning_dep * 1.5 AND evening_dep > evening_arr * 1.5 THEN 'Office'
                WHEN weekend_activity > weekday_activity * 1.2 THEN 'Leisure'
                WHEN activity_std < 2 THEN 'Low Activity'
                ELSE 'Mixed'
            END as station_type
        FROM station_characteristics
        WHERE avg_total_activity > 1  -- Filter very low activity
        ORDER BY avg_total_activity DESC
    """)
    
    print("Station Type Distribution:")
    print(profiles['station_type'].value_counts())
    
    print("\nTop 5 Residential Stations:")
    print(profiles[profiles['station_type'] == 'Residential'].head())
    
    print("\nTop 5 Office Stations:")
    print(profiles[profiles['station_type'] == 'Office'].head())
    
    return profiles

# Run the fixed analysis
if __name__ == "__main__":
    db = BikeDataDB()
    db.connect()
    
    # Run corrected analyses
    check_data_balance(db)
    patterns = analyze_station_patterns_correctly(db)
    analyze_net_flow_patterns(db)
    profiles = identify_real_station_types(db)
    
    # Save corrected profiles
    profiles.to_csv('station_profiles_fixed.csv', index=False)
    print("\n✅ Fixed analysis complete!")