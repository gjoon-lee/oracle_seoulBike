import psycopg2
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime
import logging

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'bike_data',
    'user': 'postgres',
    'password': '123Ryanlee!',
    'port': 5432
}

class BikeDataDB:
    """PostgreSQL connection manager for Seoul Bikes project"""
    
    def __init__(self):
        self.engine = None
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Create SQLAlchemy engine for pandas integration"""
        try:
            # Build connection string
            conn_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            self.engine = create_engine(conn_string)
            self.logger.info("Connected to PostgreSQL")
            return self.engine
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            raise
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                self.logger.info(f"PostgreSQL version: {version}")
                
                # Check tables
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                tables = [row[0] for row in result]
                self.logger.info(f"Tables found: {tables}")
                return True
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """Execute a raw SQL query"""
        with self.engine.connect() as conn:
            return conn.execute(text(query), params)
    
    def insert_dataframe(self, df, table_name, if_exists='append'):
        """Insert pandas DataFrame to PostgreSQL"""
        try:
            rows = df.to_sql(
                table_name, 
                self.engine, 
                if_exists=if_exists,
                index=False,
                method='multi'  # Faster bulk inserts
            )
            self.logger.info(f"Inserted {rows} rows into {table_name}")
            return rows
        except Exception as e:
            self.logger.error(f"Insert failed: {e}")
            raise
    
    def read_query(self, query):
        """Read SQL query into pandas DataFrame"""
        return pd.read_sql_query(query, self.engine)
    
    def get_station_hourly_flow(self, station_id, start_date, end_date):
        """Get hourly flow data for a specific station"""
        query = """
            SELECT * FROM station_hourly_flow
            WHERE station_id = %s 
            AND flow_date BETWEEN %s AND %s
            ORDER BY flow_date, flow_hour
        """
        return pd.read_sql_query(
            query, 
            self.engine,
            params=(station_id, start_date, end_date)
        )
    
    def log_data_quality(self, file_name, stats):
        """Log data quality metrics"""
        query = """
            INSERT INTO data_quality_log 
            (file_name, record_date, total_rows, valid_rows, 
             duplicate_rows, encoding_errors, processing_time_sec, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self.engine.connect() as conn:
            conn.execute(text(query), (
                file_name,
                stats['record_date'],
                stats['total_rows'],
                stats['valid_rows'],
                stats['duplicate_rows'],
                stats['encoding_errors'],
                stats['processing_time'],
                stats['status']
            ))

# Quick test script
if __name__ == "__main__":
    # Initialize connection
    db = BikeDataDB()
    db.connect()
    
    # Test connection
    if db.test_connection():
        print("\nPostgreSQL is ready for your Seoul Bikes project!")
        
        # Example: Check if tables exist
        tables_check = db.read_query("""
            SELECT table_name, 
                   pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        print("\nYour tables:")
        print(tables_check)
    else:
        print("\nConnection failed. Check your PostgreSQL installation.")