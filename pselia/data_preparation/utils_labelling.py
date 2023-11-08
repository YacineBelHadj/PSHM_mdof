from pselia.config_elia import load_events, load_stage
from datetime import datetime



def get_event_state_stage(dt):
    events = load_events()
    stage = load_stage()

    event_name = None
    event_state = None
    stage_name = None

    # Check which event the datetime falls into
    for event, details in events.items():
        if dt >= details['timestamp']:
            event_name = event
            event_state = details['state']
        else:
            break  # Assumes events are in chronological order and dt has reached a future event

    # Check which stage the datetime falls into
    for stage_key, period in stage.items():
        if period['start'] <= dt < period['end']:
            stage_name = stage_key
            break

    # Return a tuple of event, state, and stage
    return event_name, event_state, stage_name

