import numpy as np
import sqlite3


def load_freq_axis(database_processed_path):
    conn = sqlite3.connect(database_processed_path)
    c = conn.cursor()
    c.execute("SELECT freq FROM metadata")
    freq = c.fetchone()[0]
    freq = np.frombuffer(freq, dtype=np.float64)
    return freq