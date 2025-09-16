import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from package import agent,datap,fit,pic
from collections import defaultdict

def process_IC(agent_name, all_fitdata):
    AIC = defaultdict(list)
    BIC = defaultdict(list)
    fit_info = {}
    all_sub_info = []
    rows = [] 

    for agent_fitdata in all_fitdata.values(): 
        fit_info['bic'] = [fitdata['bic'] for fitdata in agent_fitdata]
        all_sub_info.append(fit_info) # get fit information (BIC) for calculating PXP
        for subj_id, fitdata in agent_fitdata.items():
            AIC[subj_id].append(fitdata['aic']) # get AIC
            BIC[subj_id].append(fitdata['bic']) # get BIC
    BMS_result = fit.fit.bms(all_sub_info,use_bic=True) # calculate PXP
    crs_PXP = BMS_result['pxp'] # the sort of agent name is the same as all_fitdata.keys()

    for subj_id in BIC.keys():            
        AIC_diff = [x - AIC[subj_id][0] for x in AIC[subj_id]] # mode index 0: MixedArb-Dynamic model 
        BIC_diff = [x - BIC[subj_id][0] for x in BIC[subj_id]]
        for i,name in enumerate(agent_name):
            row = {
            'agent':name,
            'subj_id': subj_id,
            'AIC': AIC_diff[i],
            'BIC': BIC_diff[i],
            'PXP': crs_PXP[i],
            }        
            rows.append(row)

    crs_table = pd.DataFrame(rows)
    return crs_table

def violin(ax, data, x, y, order = None, palette = None, orient='v',
        hue=None, hue_order=None, 
        mean_marker_size=6, err_capsize=.11, scatter_size=7):

        g_var = y if orient=='h' else x
        v_var = x if orient=='h' else y
        v=sns.violinplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=g_var if hue is None else hue, 
                            hue_order=order if hue is None else hue_order, 
                            orient=orient, palette=palette, 
                            legend=False, alpha=.1, inner=None, density_norm='width',
                            ax=ax)
        plt.setp(v.collections, alpha=.35, edgecolor='none')
        sns.stripplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=g_var if hue is None else hue, 
                            hue_order=order if hue is None else hue_order, 
                            orient=orient, palette=palette, 
                            size=scatter_size,
                            edgecolor=None, jitter=True, alpha=.7,
                            dodge=False if hue is None else True,
                            legend=False, zorder=2,
                            ax=ax)
        sns.barplot(data=data, 
                            x=x, y=y, order=order, 
                            orient=orient, 
                            hue=hue, hue_order=hue_order,
                            errorbar='sd', linewidth=1, 
                            edgecolor=(0,0,0,0), facecolor=(0,0,0,0),
                            capsize=err_capsize,
                            ax=ax)

        groupby = [g_var, hue] if hue is not None else [g_var]
        sns.stripplot(data=data.groupby(by=groupby)[v_var].mean().reset_index(), 
                        x=x, y=y, order=order, 
                        hue=hue, hue_order=hue_order, 
                        palette=[[.2]*3]*len(hue_order) if hue is not None else None,
                        dodge=False if hue is None else True,
                        marker='o', size=mean_marker_size, color=[.2]*3, ax=ax)
        ax.set(xlabel=None)

# model comparison
def viz_sortviolin(agent_name,all_fitdata):
    fig, ax = plt.subplots(1, 3, figsize=(4, 12))
    ax = ax.flatten()

    crs_table,crs_PXP = process_IC(agent_name,all_fitdata)

    for i, critics in enumerate(['AIC','BIC','PXP']):
        critics_means = crs_table.groupby('agent')[critics].mean()
        sorted_models = critics_means.sort_values(ascending=True).index.tolist()
        violin(data=crs_table, x=critics, y='agent', 
            order=sorted_models, orient='h',color=(161/255,169/255,208/255),palette=None,
            mean_marker_size=8, scatter_size=2,
            err_capsize=.1,
            ax=ax[i])
        
        if critics == 'PXP':
            sorted_indices = np.argsort(-np.array(crs_table['PXP']))
            sorted_models = [crs_table['agent'][i] for i in sorted_indices]
            
            sns.barplot(data=crs_table, x=critics, y='agent', 
                        order=sorted_models, color = (161/255,169/255,208/255),
                        orient='h', ax=ax[i])
           
        indices = [agent_name.index(x)+1 for x in sorted_models]
        ax[i].axvline(x=0, ymin=0, ymax=1, color='k', ls='--', lw=1.5)
        ax[i].set_yticks(list(range(len(sorted_models)))) 
        ax[i].set_yticklabels(indices)
        ax[i].spines['left'].set_position(('axes',-0.1))
        ax[i].set_box_aspect(1.5)
        for pos in ['bottom','left']: ax[i].spines[pos].set_linewidth(2)
        for pos in ['top','right']: ax[i].spines[pos].set_visible(False)

def viz_sortcurve(agent_dict, agents, markers, crs='BIC'): 
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


if __name__ == '__main__':
    # STEP 1: LOAD DATA 
    cfg = datap.load_config()
    dir = cfg["data_dir"]
    agent_name = ['MDT','MB','MF'] #'MB'
    
    Groups_all_fitdata = {}
    MUD_all_fitdata = {}
    HC_all_fitdata = {}  

    # STEP 2: PARAMS COMPARITION
    for name in agent_name:
        task_agent = getattr(agent,name)
        with open(f'{dir}/fitdata/fitresults_MUD_{name}.pkl', 'rb') as f: 
                MUD_fitdata = pickle.load(f)
        with open(f'{dir}/fitdata/fitresults_HC_{name}.pkl', 'rb') as f: 
                HC_fitdata = pickle.load(f)
        Groups_fitdata = HC_fitdata | MUD_fitdata
        
        Groups_all_fitdata[name] = Groups_fitdata
        MUD_all_fitdata[name] = MUD_fitdata
        HC_all_fitdata[name] = HC_fitdata

    viz_sortviolin(agent_name,Groups_all_fitdata)
    viz_sortviolin(agent_name,MUD_all_fitdata)
    viz_sortviolin(agent_name,HC_all_fitdata)
    #viz_sortcurve(task_agent,agent_fitdata)

