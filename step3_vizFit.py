import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from package import agent,datap,fit,pic
from scipy.stats import pearsonr,ttest_ind



def disp_corr(ax, data, xname, yname):
        r, p = pearsonr(data[xname], data[yname])
        print(f"{xname}: t = {r:.3f}, p = {p:.4f}")
        ax.text(
            0.05, 0.95,
            f"r = {r:.2f}\np = {p:.3g}",
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )

def disp_ttest(ax, data, y, group_col="group"):

    groups = data[group_col].unique()
    if len(groups) != 2:
        print(f"[Skip] {y}: too many groups for t-test!")
        return  

    g1, g2 = groups
    vals1 = data.loc[data[group_col] == g1, y].dropna()
    vals2 = data.loc[data[group_col] == g2, y].dropna()

    t, p = ttest_ind(vals1, vals2, equal_var=False)

    print(f"{y}: t = {t:.3f}, p = {p:.4f}")

    # 显著性标记
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = None

    if sig:
        ymax = max(max(vals1), max(vals2))
        y_line = ymax * 1.1  
        ax.plot([0.3, 0.7], [y_line, y_line], color='black', linewidth=1.2)  
        ax.text(0.5, y_line*1.02, sig, ha='center', va='bottom', fontsize=12)

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

def viz_params(agent,agent_fitdata):
    model_name = agent.name
    params_name = agent.p_name
    params_num = len(params_name)
    ncols = 3
    nrows = int(np.ceil(params_num / ncols))    
    palette = [[68/255,97/255,123/255],[255/255,85/255,84/255]]

    print(f'Displaying params comparision of model: "{model_name}"')
    fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    ax = ax.flatten()

    for i,p_name in enumerate(params_name): 
        violin(ax[i], agent_fitdata, "group", p_name, palette=palette)

        disp_ttest(ax[i], agent_fitdata, y=p_name)

        pic.set_format(ax[i], 'Group', p_name)

        

    for j in range(i+1, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()
    plt.show()

def viz_recov(agent,agent_rawdata,agent_fitdata):
    agent_alldata = datap.load_data(agent,agent_rawdata,agent_fitdata)
    model_name = agent.name
    params_name = agent.p_name
    params_num = len(params_name)
    param_bnds = agent.bnds
    ncols = 3
    nrows = int(np.ceil(params_num / ncols))    

    print(f'Displaying params comparision of model: "{model_name}"')
    fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    ax = ax.flatten()

    for i,p_name in enumerate(params_name): 
        rp_name = "r_" + p_name 

        sns.scatterplot(
                data=agent_alldata, 
                x= p_name , y= rp_name,
                marker= 'o',s=20, alpha=.8, edgecolor='none',
                clip_on=False,ax = ax[i])

        disp_corr(ax[i], agent_alldata, p_name, rp_name)

        p_bnds = param_bnds[i]
        pic.set_ticks(agent_alldata[[rp_name,p_name]], ax=ax[i], 
                      lb=p_bnds[0], ub=p_bnds[1], whichaxis='x', 
                      ticksnum=5, precision=0.5)
        pic.set_ticks(agent_alldata[[rp_name,p_name]], ax=ax[i], 
                      lb=p_bnds[0], ub=p_bnds[1], whichaxis='y', 
                      ticksnum=5, precision=0.5)
        pic.set_format(ax[i], p_name, rp_name)

    # 删除多余的空子图
    for j in range(i+1, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    ## STEP 1: LOAD DATA 
    cfg = datap.load_config()
    dir = cfg["data_dir"]
    agent_name = ['MDT','MB','MF'] #'MB'
    agent_dict = {}

    # STEP 2: PARAMS COMPARITION
    # for name in agent_name:
    #     task_agent = getattr(agent,name)
    #     with open(f'{dir}/fitdata/fitresults_MUD_{name}.pkl', 'rb') as f: 
    #             MUD_fitdata = pickle.load(f)

    #     with open(f'{dir}/fitdata/fitresults_HC_{name}.pkl', 'rb') as f: 
    #             HC_fitdata = pickle.load(f)

    #     agent_fitdata = HC_fitdata | MUD_fitdata;  
    #     agent_fitdata = datap.load_data(task_agent,agent_fitdata)
    #     viz_params(task_agent,agent_fitdata)

    # STEP 2: PARAM RECOVERY 
    for name in agent_name:
        task_agent = getattr(agent,name)
        with open(f'{dir}/{name}_alldata.pkl', 'rb') as f: 
            agent_rawdata = pickle.load(f)
        with open(f'{dir}/fitdata/fitresults_sim_{name}.pkl', 'rb') as f: 
            agent_fitdata = pickle.load(f)
        viz_recov(task_agent,agent_fitdata,agent_rawdata)
