import numpy as np
import sqlite3
from pselia.config_elia import settings


def load_freq_axis(database_processed_path):
    conn = sqlite3.connect(database_processed_path)
    c = conn.cursor()
    c.execute("SELECT freq FROM metadata")
    freq = c.fetchone()[0]
    freq = np.frombuffer(freq, dtype=np.float64)
    return freq

# a function that takes the event and returns the abbr
def get_event_to_abbr():
    event_to_abbr = dict()
    for event_key in settings.event.keys():
        event_abbr =getattr(settings.event,event_key).abbr
        event_to_abbr[event_key] = event_abbr
    return event_to_abbr
def get_sorted_event():
    # Convert Dynaconf settings to a regular dictionary
    events = settings.EVENT.to_dict()
    # Now proceed with sorting the events based on timestamp
    sorted_events = sorted(
        ((key, value) for key, value in events.items() if 'timestamp' in value),
        key=lambda x: x[1]['timestamp']
    )
    # Return a list of event abbreviations in sorted order
    return [event[1]['name'] for event in sorted_events]

    
    