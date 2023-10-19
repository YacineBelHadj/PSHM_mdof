from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

def compute_auc(anomaly,healthy):
    """
    Function to compute AUC-ROC from healthy and anomalous scores.

    Parameters
    ----------
    healthy : ndarray
        Array of 'score_column' scores for healthy instances.
    anomaly : ndarray
        Array of 'score_column' scores for anomalous instances.

    Returns
    -------
    float
        The computed AUC-ROC value.
    """
    # Create binary labels: 0 for healthy and 1 for anomaly
    y_true = np.concatenate([np.zeros_like(healthy), np.ones_like(anomaly)])

    # Concatenate the scores
    y_score = np.concatenate([healthy, anomaly])

    # Compute and return AUC-ROC score
    return roc_auc_score(y_true, y_score)


def compute_auc_for_levels(df,score_column='log_likelihood',scaling_factor=-1):
    """
    Function to compute AUC-ROC for each system and anomaly level.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame, which includes 'system_name', 'anomaly_level', and 'score_column' columns.

    Returns
    -------
    DataFrame
        A DataFrame where each row corresponds to a unique combination of 'system_name' and 'anomaly_level',
        and the 'auc' column contains the corresponding AUC-ROC score.
    """
    systems = df['system_name'].unique()
    anomaly_levels = df['anomaly_level'].unique()

    auc_dict = {'system_name': [], 'anomaly_level': [], 'auc': []}

    for system in systems:
        system_df = df[df['system_name'] == system]
        healthy = scaling_factor*system_df[system_df['stage'] == 'test'][score_column].values
        # Negating 'log_likelihood' values because higher values indicate more anomaly

        for level in anomaly_levels:
            anomaly = scaling_factor*system_df[system_df['anomaly_level'] == level][score_column].values

            # Compute AUC using the defined function
            auc = compute_auc(healthy=healthy, anomaly=anomaly)

            auc_dict['system_name'].append(system)
            auc_dict['anomaly_level'].append(level)
            auc_dict['auc'].append(auc)

    # Convert the dictionary to DataFrame and return
    auc_results_df = pd.DataFrame(auc_dict)
    auc_results_df=auc_results_df.pivot(index='system_name', columns='anomaly_level', values='auc')
    # change the index from 0 ==> system_0
    auc_results_df.index = [f'system_{i}' for i in auc_results_df.index]

    return auc_results_df
