import pickle,os
import numpy as np
from package import agent,datap,env



if __name__ == '__main__':
    #设置参数，生成数据
    seed = 9809
    cfg = datap.load_config()
    dir = cfg["data_dir"]
    sim_mode = cfg["sim_mode"]

    task_env = env.two_stage() 
    agent_name = ['Hybrid','MDT','MB','MF'] #'MDT','MB','MF','RA'

    turns = 200

    if sim_mode == 'init sim': 
        for name in agent_name:
            agent_data = {}
            task_agent = getattr(agent,name)
            for t in range(turns):
                params, simdata = datap.block(task_agent,task_env,seed) 
                dataname = f'{name}simdata_{t+1}'
                simdata.to_excel(f'{dir}/simdata/{name}/{dataname}.xlsx',index=False)
                agent_data[dataname] = [params, simdata]
                seed +=1

            with open(f'{dir}/behavdata_{name}_sim.pkl', 'xb') as f:    
                pickle.dump(agent_data, f)
    
    elif sim_mode == 'fit sim':
        groups = ['MUD','HC']
        for group in groups:
            for name in agent_name:
                task_agent = getattr(agent,name)
                agent_data = {}
                with open(f'{dir}/fitdata/fitresults_{name}_{group}.pkl', 'rb') as f: 
                    group_fitdata = pickle.load(f)
                for subj, fitdata in group_fitdata.items():
                    init_params = fitdata['param']
                    params, simdata = datap.block(task_agent,task_env,seed,init_params,None) 
                    simdata.to_excel(f'{dir}/simdata/{name}/{name}_{subj}_simbyfit.xlsx',index=False)
                    agent_data[subj] = simdata

                with open(f'{dir}/behavdata_{name}_{group}fit.pkl', 'xb') as f:    
                    pickle.dump(agent_data, f)

    elif sim_mode == 'MDT walk':
        group_files = {"MUD": f"{dir}/behavdata_MUD_true.pkl",
                        "HC": f"{dir}/behavdata_HC_true.pkl"}
        
        for group, pkl_path in group_files.items():
            true_data = datap.load_pkl(os.path.dirname(pkl_path), os.path.splitext(os.path.basename(pkl_path))[0])
            agent_data = {}
            with open(f'{dir}/fitdata/fitresults_MDT_{group}.pkl', 'rb') as f: 
                    group_fitdata = pickle.load(f)
            task_agent = getattr(agent, 'MDT')

            for subj, fitdata in group_fitdata.items():
                    init_params = fitdata['param']
                    params, simdata = datap.block(task_agent,task_env,seed,init_params,true_data[subj]) 
                    simdata.to_excel(f'{dir}/simdata/MDT/MDT_{subj}_walkbyfit.xlsx',index=False)
                    agent_data[subj] = simdata

            with open(f'{dir}/behavdata_MDT_{group}walk.pkl', 'xb') as f:    
                    pickle.dump(agent_data, f)

    print('Done')



