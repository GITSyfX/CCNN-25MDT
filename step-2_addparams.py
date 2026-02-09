import pandas as pd
import pickle
from package import agent,datap

cfg = datap.load_config()
dir = cfg["data_dir"]
agent_name = ['Hybrid','MDT','MB'] 
agent_dict = {}
params_name = ['omega','A_F2B','beta']

datafile = 'D:/CCNN/Projects/MDT/Data/alldata_summary.xlsx' 
savefile = 'D:/CCNN/Projects/MDT/Data/allfitdata_summary.xlsx' 
 
df = pd.read_excel(datafile)
print(f"读取Excel文件，共 {len(df)} 行数据")

# 读取pkl文件

for i,a_name in enumerate(agent_name):
    p_name = params_name[i] 
    task_agent = getattr(agent,a_name)
    with open(f'{dir}/fitdata/fitresults_{a_name}_MUD.pkl', 'rb') as f: 
            MUD_fitdata = pickle.load(f)

    with open(f'{dir}/fitdata/fitresults_{a_name}_HC.pkl', 'rb') as f: 
            HC_fitdata = pickle.load(f)

    agent_fitdata = HC_fitdata | MUD_fitdata;  
    agent_fitdata = datap.load_data(task_agent,agent_fitdata)
    df[f'{a_name} {p_name}'] = agent_fitdata[f'{p_name}']

# ========== 提取参数并添加到DataFrame ==========

# ========== 保存结果 ==========
df.to_excel(savefile, index=False)
print(f"结果已保存到 {savefile}")