import sqlite3

# Create a database file
conn = sqlite3.connect('seoul_bike.db')
cursor = conn.cursor()

# Creating bike_availability table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS bike_availability (
        station_id TEXT,
        timestamp DATETIME,
        available_bikes INTEGER,
        available_racks INTEGER,
        PRIMARY KEY (station_id, timestamp)
    )
''')
conn.commit()

# Creating stations table for coordinates
cursor.execute('''
    CREATE TABLE IF NOT EXISTS stations (
        station_id TEXT PRIMARY KEY,
        station_name TEXT,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')