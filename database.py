import sqlite3
from datetime import datetime


DATABASE_NAME = "safety.db"


def create_database():
    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            worker_id INTEGER,
            missing_ppe TEXT NOT NULL,
            image_path TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_violation(worker_id, missing_ppe, image_path):
    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO violations
        (timestamp, worker_id, missing_ppe, image_path)
        VALUES (?, ?, ?, ?)
    """, (
        timestamp,
        worker_id,
        ", ".join(missing_ppe),
        image_path
    ))

    connection.commit()
    connection.close()


def get_violations():
    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, timestamp, worker_id,
               missing_ppe, image_path
        FROM violations
        ORDER BY id DESC
    """)

    violations = cursor.fetchall()

    connection.close()

    return violations


if __name__ == "__main__":
    create_database()

    print("✅ Safety database created successfully.")