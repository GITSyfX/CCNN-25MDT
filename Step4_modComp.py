import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from package import agent,datap,fit
from scipy.stats import pearsonr

def get_crs(results):
    AIC_means = []
    BIC_means = [] 
    AIC_se = [] 
    BIC_se = [] 
    AIC_raw = []
    BIC_raw = []
    AIC = np.array([result['aic'] for result in results])
    BIC = np.array([result['bic'] for result in results])
    AIC_means = np.mean(AIC)
    BIC_means = np.mean(BIC)
    AIC_se = np.std(AIC,ddof=1)/np.sqrt(len(AIC))
    BIC_se = np.std(BIC,ddof=1)/np.sqrt(len(BIC))
    AIC_raw = AIC
    BIC_raw = BIC
    return AIC_means,BIC_means,AIC_se,BIC_se,AIC_raw,BIC_raw 

def get_pxp(agent_dict):
    all_sub_info = []
    fit_info = {}
    for name in agent_dict.keys():
        results = agent_dict[name]
        fit_info['bic'] = [result['bic'] for result in results]
        all_sub_info.append(fit_info)
    BMS_result = fit.fit.bms(all_sub_info,use_bic=True)
    return BMS_result['pxp']

def viz_crscur(agent_dict, agents, markers, crs='BIC'): 
    sel_table = pd.DataFrame.from_dict(agent_dict.copy())
    sel_table[f'min_{crs}'] = sel_table.apply(
        lambda x: np.min([x[f'{crs}_{agent.name}'] for agent in agents]), 
        axis=1)
    sort_table = sel_table.sort_values(by=f'min_{crs}').reset_index()
    sort_table['sub_seq'] = sort_table.index
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))

    for i, agent in enumerate(agents):
        marker = markers[i]
        sns.scatterplot(x='sub_seq', y=f'{crs}_{agent.name}', 
                        data=sort_table, label = agent.name,
                        marker= marker, 
                        s=20, alpha=.8,
                        edgecolor='none', ax=ax)
    ax.set_xlim([-5, sort_table.shape[0]+15])
    ax.legend(loc='upper left')
    ax.spines['left'].set_position(('axes',-0.02))
    ax.set_xlabel(f'Participant index\n(sorted by the minimum {crs} score over all models)')
    ax.set_ylabel(crs.upper())
    fig.tight_layout()
