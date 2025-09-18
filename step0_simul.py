import pickle,os
from package import agent,datap,env



if __name__ == '__main__':
    #设置参数，生成数据
    seed = 2025
    cfg = datap.load_config()
    dir = cfg["data_dir"]
    sim_mode = cfg["sim_mode"]

    task_env = env.two_stage() 
    agent_name = ['MDT','MB','MF','RA'] #'MB','MF','RA'

    turns = 200

    if sim_mode == 'init sim': 
        for name in agent_name:
            agent_data = {}
            task_agent = getattr(agent,name)
            for t in range(turns):
                params, simdata = datap.block(task_agent,task_env,[],seed) 
                dataname = f'{name}simdata_{t+1}'
                simdata.to_excel(f'{dir}/simdata/{name}/{dataname}.xlsx',index=False)
                agent_data[dataname] = [params, simdata]
                seed +=1

            with open(f'{dir}/{name}_alldata.pkl', 'xb') as f:    
                pickle.dump(agent_data, f)
    
    elif sim_mode == 'fit sim':
        groups = ['MUD','HC']
        for group in groups:
            for name in agent_name:
                task_agent = getattr(agent,name)
                agent_data = {}
                with open(f'{dir}/fitdata/fitresults_{group}_{name}.pkl', 'rb') as f: 
                    group_fitdata = pickle.load(f)
                for subj, fitdata in group_fitdata.items():
                    init_params = fitdata['param']
                    params, simdata = datap.block(task_agent,task_env,init_params,seed) 
                    simdata.to_excel(f'{dir}/simdata/{name}/{name}_{subj}_simbyfit.xlsx',index=False)
                    agent_data[subj] = simdata

                with open(f'{dir}/{name}_{group}_alldata.pkl', 'xb') as f:    
                    pickle.dump(agent_data, f)

    elif sim_mode == 'MDT walk':
        group_files = {"MUD": f"{dir}/Behavdata_MUD_true.pkl",
                        "HC": f"{dir}/Behavdata_HC_true.pkl"}
        
        for group, pkl_path in group_files.items():
            true_data = datap.load_pkl(os.path.dirname(pkl_path), os.path.splitext(os.path.basename(pkl_path))[0])
            agent_data = {}
            with open(f'{dir}/fitdata/fitresults_{group}_MDT.pkl', 'rb') as f: 
                    group_fitdata = pickle.load(f)
            task_agent = getattr(agent, 'MDT')

            for subj, fitdata in group_fitdata.items():
                    init_params = fitdata['param']
                    params, simdata = datap.block(task_agent,task_env,init_params,seed,true_data[subj]) 
                    simdata.to_excel(f'{dir}/simdata/MDT/MDT_{subj}_walkbyfit.xlsx',index=False)
                    agent_data[subj] = simdata

            with open(f'{dir}/MDT_{group}walk_alldata.pkl', 'xb') as f:    
                    pickle.dump(agent_data, f)

    print('Done')



