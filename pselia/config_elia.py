from dynaconf import Dynaconf
from datetime import datetime
import pprint
import os 
from pathlib import Path 

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent
print(ROOT_PATH)


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['configuration/settings_elia.toml',
                    'configuration/.secrets_elia.toml'],
    root_path= ROOT_PATH
)
# check the path used by dynaconf to load the settings file

def load_processed_data_path(settings_name:str):
    root_processed_data_path = Path(settings.dataelia.path['processed'])
    processed_data_path = (root_processed_data_path/settings_name.lower()).with_suffix('.db')
    return processed_data_path

def load_processed_data_path_vas(settings_name:str):
    root_processed_data_path = Path(settings.dataelia.path['processed'])
    processed_data_path = (root_processed_data_path/(settings_name.lower()+'_vas')).with_suffix('.db')
    return processed_data_path

def get_data_path(settings_name:str, data_type:str):
    if data_type == 'psd_original':
        return load_processed_data_path(settings_name)
    elif data_type == 'psd_notch':
        return load_processed_data_path_vas(settings_name)
    else:
        raise ValueError(f"data_type {data_type} not supported")

def load_events():
    events = {}
    for event_name, event_data in settings.EVENT.items():
        timestamp = datetime.strptime(event_data['timestamp'], '%Y-%m-%d %H:%M:%S')
        state = event_data['state']
        events[event_name] = {'timestamp': timestamp, 'state': state}
    return events

def load_stage():
    stages = {}
    for stage_name, stage_data in settings.STAGE.items():
        start = datetime.strptime(stage_data['start'], '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(stage_data['end'], '%Y-%m-%d %H:%M:%S')
        stages[stage_name] = {'start': start, 'end': end}
    return stages

def load_activities():
    activities = {}
    for activity_name, activity_data in settings.ACTIVITY.items():
        start = datetime.strptime(activity_data['start']['timestamp'], '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(activity_data['end']['timestamp'], '%Y-%m-%d %H:%M:%S')
        activities[activity_name] = {'start': start, 'end': end}
    return activities

def load_measurement_bound():
    t_s = settings.measurements['start']
    t_e = settings.measurements['end']
    start = datetime.strptime(t_s, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(t_e, '%Y-%m-%d %H:%M:%S')
    return start, end

def load_optimazation_path():
    root_optimization_path = Path(settings.dataelia.path.optimization_optuna)
    return root_optimization_path



if __name__ == "__main__":
    events = load_events()
    stages = load_stage()
    activities = load_activities()

    pprint.pprint(events)
    print("\n")
    pprint.pprint(stages)
    print("\n")
    pprint.pprint(activities)
