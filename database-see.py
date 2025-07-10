import sqlite3
import os

# Path to database inside instance folder
db_path = os.path.join('instance', 'users.db')

# Connect to the correct database file
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

# View users (use lowercase table names)
print("\nUsers:")
try:
    for row in cursor.execute("SELECT * FROM user"):
        print(row)
except sqlite3.OperationalError as e:
    print("Error fetching users:", e)

# View predictions
print("\nPredictions:")
try:
    for row in cursor.execute("SELECT * FROM prediction"):
        print(row)
except sqlite3.OperationalError as e:
    print("Error fetching predictions:", e)

conn.close()
