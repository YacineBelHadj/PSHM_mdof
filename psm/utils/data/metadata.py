import sqlite3
import numpy as np
import sqlite3
import numpy as np
# load the database path
from config import settings
from pathlib import Path



def get_metadata_processed(settings_proc, settings_simu):
    root= Path(settings.data.path["processed"])
    database_path = (root /settings_proc/settings_simu.lower()).with_suffix('.db')
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute("SELECT * FROM metadata")
    metadata = c.fetchone()
    fs, nperseg, filter_order, lpf, freq, SNR = metadata
    freq_axis = np.frombuffer(freq, dtype=np.float64)
    res = {"fs":fs, "nperseg":nperseg, "filter_order":filter_order, "lpf":lpf, "freq":freq_axis, "SNR":SNR}
    # close the connection
    conn.close()
    return res