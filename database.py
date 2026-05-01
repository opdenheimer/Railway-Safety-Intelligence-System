import sqlite3
import datetime

DB_NAME = 'railway_safety.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            incident_type TEXT NOT NULL,
            track_id INTEGER,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_incident(incident_type, track_id, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO incidents (timestamp, incident_type, track_id, confidence)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, incident_type, track_id, confidence))
    conn.commit()
    conn.close()

def get_recent_incidents(limit=20):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, incident_type, track_id, confidence
        FROM incidents
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == '__main__':
    init_db()
    print("Database initialized.")
