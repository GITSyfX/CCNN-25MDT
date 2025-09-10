import pickle
from package import agent,datap,env


#设置参数，生成数据
seed = 2025
cfg = datap.load_config()
dir = cfg["data_dir"]
task_env = env.two_stage() 
agent_name = ['MDT'] #'MB','MF','RA'

turns = 200
for name in agent_name:
    agent_data = {}
    for t in range(turns):
        task_agent = getattr(agent,name)
        params, simdata = datap.block(task_agent,task_env,[],seed) 
        dataname = f'{name}simdata_{t+1}'
        simdata.to_excel(f'{dir}/simdata/{name}/{dataname}.xlsx',index=False)
        agent_data[dataname] = [params, simdata]
        seed +=1

    with open(f'{dir}/{name}_alldata.pkl', 'xb') as f:    
        pickle.dump(agent_data, f)
  
print('Done')



