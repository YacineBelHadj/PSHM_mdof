"""In this file we generate the dataset resulting from the simulation of the population of systems.
there is N systems in the population, each system is simulated for T time steps.
the parameters of the simulation are saved in configurations/simulation.toml and accessed with dynaconf logic

the dataset is saved in data/raw/dataset1.db here we use sqlite3 to store the data
the dataset is composed of 3 tables:
    - systems: contains the parameters of the systems
    - simulation: contains the parameters of the simulation
    - results: contains the results of the simulation

"""
from psm.simulation.simulator import Simulator
from psm.simulation.population_manipulator import PopulationManipulator
from psm.simulation.population import Population
from psm.simulation.features import resonance_frequency_computation as res_comp
import numpy as np
from tqdm import tqdm
import logging 
import sqlite3
from config import settings
from pathlib import Path
import argparse
from functools import partial
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='logs/generate_dataset.log',
                    filemode='w')
argparse = argparse.ArgumentParser()
argparse.add_argument('--settings', type=str, default='SETTINGS1')
args = argparse.parse_args()

simu_params = settings.simulation[args.settings]['simu_params']
env_exc_params = settings.simulation[args.settings]['env_exc_params']
pop_params_path = Path(settings.simulation[args.settings]['population_param'])
anomaly_loc = settings.simulation[args.settings]['anomaly_loc']
exc_location =env_exc_params['exc_location']
path_data = Path(settings.data.path['raw'])/args.settings

def create_table(c):
    c.execute("""CREATE TABLE systems (
              id INTEGER PRIMARY KEY,
              name TEXT,
              mass BLOB, stiffness BLOB, damping BLOB,)
            """)
    c.execute(""" 
             CREATE TABLE simulation 
             (id INTEGER PRIMARY KEY,
             system_id INTEGER,
             name TEXT,
             latent REAL,
             amplitude REAL,
             anomaly_level REAL,
             resonance_frequency BLOB,
             TDD_input BLOB,
             TDD_output BLOB,
             FOREIGN KEY(system_id) REFERENCES systems(id))
             """)
    c.execute("""
            CREATE TABLE metadata
            (dt REAL, t_end REAL, 
            std_latent REAL , latent_value REAL,
            input_location REAL,
            anomaly_location REAL)
             """)

def amplitude_excitation(amp:float, loc:float, shape:float):
    # weibull distribution excitation 
    exc = amp * (-np.log(np.random.uniform(0,1)))**(1/shape) + loc
    return exc
def latent_var(mean:float, std:float):
    # latent variable is a gaussian distribution
    latent = np.random.normal(mean, std)
    return latent
def create_list_anomaly_level():
    anomaly_levels = [0]*1200 + [i/100 for i in range(1, 14,2) for _ in range(200)]
    return anomaly_levels
amp, loc,shape = env_exc_params['exc_amp'], env_exc_params['exc_loc'], env_exc_params['exc_shape']
lat_mean , lat_std = env_exc_params['lat_mu'], env_exc_params['lat_std']

get_amplitude = partial(amplitude_excitation, amp, loc, shape)
get_latent = partial(latent_var, lat_mean, lat_std)



   

def main():
    population = Population()
    population.load_population(pop_params_path)
    logging.info(f'Population loaded from {pop_params_path}')
    population_mani = PopulationManipulator(population)
    
    # create the database 
    path_data.mkdir(parents=True, exist_ok=True)

    db_path = path_data/f'{args.settings.lower()}.db'
    anomaly_levels = create_list_anomaly_level()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    if db_path.exists():
        logging.warning(f'Database already exists at {db_path}')
        # ask the user if he wants to overwrite the database
        answer = input('Do you want to overwrite the database ? (y/n)')
        if answer.lower() != 'y':
            # user chose not to overwrite, close connection and exit script
            conn.close()
            logging.info('Exiting without overwriting existing database.')
            exit()

        # delete the existing database
        db_path.unlink()
        
    # at this point, we're either working with a new database or overwriting an existing one
    create_table(c)
    logging.info(f'Database created at {db_path}')

    for a in tqdm(anomaly_levels):
        amp = get_amplitude()
        lat = get_latent()
        manipulation = [{'type':'environment','latent_value':lat,'coefficients':'load'}]
        if a != 0:
            manipulation.append({'type':'anomaly', 'location': anomaly_loc,\
                                  'anomaly_type': 'stiffness', 'anomaly_size': a})
            
        population_affected = population_mani.affect(manipulation)
        simulator = Simulator(population_affected, **simu_params)
        sim_data = simulator.simulation_white_noise(location=exc_location, amplitude=amp)
        # save the data in the database
        res_freq = res_comp(population_affected)
        systems_names = population_affected.systems_params.keys()

        systems_insert_data = []
        simulation_insert_data = []
        for name in systems_names:
            mass = population_affected.systems_params[name]['mass'].tobytes()
            stiffness = population_affected.systems_params[name]['stiffness'].tobytes()
            damping = population_affected.systems_params[name]['damping'].tobytes()
            input_data = sim_data[name]['input'].tobytes()
            output_data = sim_data[name]['output'].tobytes()
            res_freq_system = res_freq[name].tobytes()

            systems_insert_data.append((name, mass, stiffness, damping, lat, amp))
            simulation_insert_data.append((name,lat,amp,a, res_freq_system, input_data, output_data))
        
        c.executemany("""INSERT INTO systems (name, mass, stiffness, damping, latent, amplitude)
                        VALUES (?, ?, ?, ?, ?, ?)""", systems_insert_data)

        c.executemany("""INSERT INTO simulation (name,latent,amplitude, anomaly_level, resonance_frequency, TDD_input, TDD_output)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""", simulation_insert_data)

        conn.commit()
    # save the metadata
    c.execute("""INSERT INTO metadata (dt, t_end, std_latent, latent_value, input_location, anomaly_location)
                  VALUES (?, ?, ?, ?, ?, ?)""", (simu_params['dt'], simu_params['t_end'], lat_std, lat_mean, exc_location, anomaly_loc))           

if __name__ == '__main__':
    main()   
