import sqlite3

# Create a database file
conn = sqlite3.connect('seoul_bike.db')
cursor = conn.cursor()

# Create your first table
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