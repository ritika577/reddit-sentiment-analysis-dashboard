import psycopg2
import os
from dotenv import load_dotenv
# load_dotenv()

def get_db_connection():
    try:
        connection = psycopg2.connect(
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            host=os.getenv("HOST"),
            port=os.getenv("PORT"),
            dbname=os.getenv("DBNAME")
        )
        return connection
    except Exception as e:
        print(f"❌ Failed to connect to the database: {e}")
        return None

