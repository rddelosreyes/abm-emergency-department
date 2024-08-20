import multiprocessing as mp
import numpy as np
import os
import time

from environment import EDModel
from tqdm import tqdm

def single_sample(params):
    seed = os.getpid() + np.random.randint(0, 2**31)
    np.random.seed(seed)

    params['seed'] = seed

    model = EDModel(**params)
    for step_idx in range(params['max_steps']):
        model.step()

    return model.datacollector.get_model_vars_dataframe()

def run_experiment(params):
    no_iterations = params['no_iteration']

    with mp.Pool(mp.cpu_count()-4) as pool:
        args = [params] * no_iterations
        results = list(tqdm(pool.imap(single_sample, args), total=no_iterations, desc='iteration'))

    return results