import sqlite3
from pathlib import Path
from pselia.config_elia import settings, load_processed_data_path
from pselia.utils import load_freq_axis

database_path = load_processed_data_path('SETTINGS1')
def check_for_none_values(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # List of tables to check. Modify as per your database schema.
    tables_to_check = ["processed_data", "metadata"]

    for table in tables_to_check:
        # Fetch the column names for the table
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [info[1] for info in cursor.fetchall()]

        # Query to select all data from the table
        cursor.execute(f"SELECT * FROM {table};")
        rows = cursor.fetchall()

        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                if value is None:
                    print(f"None value found in table '{table}', column '{columns[col_index]}', row {row_index+1}")

    conn.close()

if __name__ == "__main__":
    check_for_none_values(database_path)
