from config import settings
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

def get_res_freq(simu_settings='SETTINGS1'):
    # Connect to the database
    database_path = (Path((settings.data.path['raw'])) / simu_settings / simu_settings.lower()).with_suffix('.db')
    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    # Execute the SQL query
    c.execute("""SELECT resonance_frequency, name FROM simulation WHERE STAGE = 'train'""")
    data = c.fetchall()

    # Extract the resonant frequencies and system names from the data
    res_freq = np.array([np.frombuffer(i[0], dtype=np.float64) for i in data])
    system_names = np.array([i[1] for i in data])

    # Convert the arrays of resonant frequencies to a DataFrame
    df_res_freq = pd.DataFrame(res_freq, columns=[f'res_freq{i+1}' for i in range(res_freq.shape[1])])

    # Add the system names as a new column
    df_res_freq['system_name'] = system_names

    # Rearrange the columns
    df_res_freq = df_res_freq[['system_name'] + [f'res_freq{i+1}' for i in range(res_freq.shape[1])]]

    # Compute the mean resonant frequency for each system
    df_res_freq = df_res_freq.groupby('system_name').mean()

    # Close the connection
    conn.close()

    return df_res_freq
