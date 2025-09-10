import pickle
import numpy as np 
import multiprocessing as mp
from package import agent,datap,env,fit

eps_ = 1e-13


def get_pool(n_fit,n_cores):
    n = n_fit
    n_cores = n_cores if n_cores else int(mp.cpu_count()*.5) 
    print(f'Using {n_cores} parallel CPU cores\{n} ')
    return mp.Pool(n_cores)


if __name__ == '__main__':
    ## STEP 0: GET PARALLEL POOL
    n_fits = 80
    n_cores = 40
    mp.freeze_support()
    pool = get_pool(n_fits,n_cores)

    ## STEP 1: LOAD 
    cfg = datap.load_config()
    dir = cfg["data_dir"]
    agent_name = ['MDT'] #'MB','MF','RA'

    ## STEP 2: SETTING 
    seed = 2025
    rng = np.random.RandomState(seed)
    task_env = env.two_stage()  

    ## STEP 3: FIT
    for name in agent_name:
        task_agent = getattr(agent,name)
        dataname = f'{name}_alldata'
        agent_data = datap.load_pkl(dir,dataname)
        results = fit.fl(pool,task_agent,agent_data,task_env,n_fits)
        with open(f'{dir}/fitdata/fitresults_sim_{name}.pkl', 'xb') as f:    
            pickle.dump(results, f)

    # summary the mean and std for parameters 
    pool.close()