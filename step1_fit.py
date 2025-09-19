import os
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
    agent_name = ['MDT'] #'MB','MF',RA'

    group_files = {
        "MUD": f"{dir}/behavdata_MUD_true.pkl",
        "HC": f"{dir}/behavdata_HC_true.pkl"
    }

    ## STEP 2: SETTING 
    seed = 2025
    rng = np.random.RandomState(seed)
    task_env = env.two_stage()  

    ## STEP 3: FIT
    for group_name, pkl_path in group_files.items():
        agent_data = datap.load_pkl(os.path.dirname(pkl_path), os.path.splitext(os.path.basename(pkl_path))[0])
        for name in agent_name:
            task_agent = getattr(agent, name)
            all_results = fit.fl(pool,task_agent,agent_data,task_env,n_fits)


            output_path = f"{dir}/fitdata/fitresults_{name}_{group_name}.pkl"
            with open(output_path, 'xb') as f:
                pickle.dump(all_results, f)

            
            print(f"已保存 {group_name} 组 {name} 模型结果 -> {output_path}")

    # summary the mean and std for parameters 
    pool.close()