import numpy as np  
import matplotlib.pyplot as plt
from package import datap


class viz_param:
    def __init__(self,indic,lim = [0,1],ticks = [0,0.5,1]):
        self.indic = eval(indic)
        self.lim = lim
        self.ticks = ticks

def basic_format(ax, x_name, y_name):
    font = {'family': 'Arial', 'weight': 'regular'}
    plt.rc('font', **font)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.tick_params(axis='both', which='major', width=1)
    ax.tick_params(axis='both', which='minor', width=1)

    # 去除边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel(y_name,fontsize=10, fontdict=font)
    plt.yticks(fontsize=10)
    ax.set_xlabel(x_name, fontsize=10, fontdict=font)
    plt.xticks(fontsize=10)  
    

def get_curvedata(agent_data,indic):
    "data of all subjects is loaded in dict  forms"
    ind_arr = np.zeros(150) #size of ind_arr = task trials

    for i,subjname in enumerate(agent_data.keys()): 
        data = agent_data[subjname]
        ind = indic(data)
        ind_arr = np.sum((ind_arr,ind[0]), axis=0) if i==0 else np.vstack((ind_arr,ind[0]))

    se = np.zeros(150) if i==0 else np.std(ind_arr, axis=0, ddof=1)/np.sqrt(np.size(ind_arr,0))
    avg =  ind_arr if i==0 else np.mean(ind_arr, axis=0)
    l_bound = avg - se
    u_bound = avg + se
    return avg,l_bound,u_bound


def viz_curve(data_dict,data_name,data_color,indic,indicname,y_lim = [0,1],y_ticks = [0,0.5,1]):
    print(f'Proccessing data of {indicname}')
    trials = np.arange(1,151,1)
    fig,ax = plt.subplots()
    ax.set_xlabel('Trials')
    ax.set_ylabel(indicname)
    
    for i,data_allsubj in enumerate(data_dict.keys()):
        avg,l_bound,u_bound = get_curvedata(data_dict[data_allsubj],indic)
        ax.plot(trials,avg, color = (data_color[i]), label = data_name[i])
        ax.fill_between(trials, l_bound, u_bound, color = (data_color[i]), alpha=0.2)
    x_tick_hightlight = [37,75,112]
    for x_tick in x_tick_hightlight:
        ax.axvline(x=x_tick,color = (87/255,96/255,105/255), linestyle='--',alpha=0.2)
    
    ax.set_ylim(y_lim[0],y_lim[1])
    ax.set_yticks(y_ticks)
    ax.set_xlim(0,150)
    ax.set_xticks([0,37,75,112,150])
    ax.text(0.105, 0.1, 'HS',transform=ax.transAxes,fontsize=12, alpha = 0.5)
    ax.text(0.355, 0.1, 'HF',transform=ax.transAxes, fontsize=12, alpha = 0.5)
    ax.text(0.605, 0.1, 'LS',transform=ax.transAxes, fontsize=12, alpha = 0.5)
    ax.text(0.855, 0.1, 'LF',transform=ax.transAxes, fontsize=12, alpha = 0.5)
    basic_format(ax,'Trials',indicname)
    fig.legend(frameon=False)
    plt.show()


def get_bardata(agent_data,indic):
    ind_HS = np.empty(0)
    ind_HF = np.empty(0)
    ind_LS = np.empty(0)
    ind_LF = np.empty(0)
    i=1
    for subjname in agent_data.keys():
        #plt.plot(trials,avg_corr_simdata)
        data = agent_data[subjname]
        *_,H_uncer_spe,H_uncer_flex,L_uncer_spe,L_uncer_flex = indic(data)
        ind_HS = np.append(ind_HS,H_uncer_spe)
        ind_HF = np.append(ind_HF,H_uncer_flex)
        ind_LS = np.append(ind_LS,L_uncer_spe)
        ind_LF = np.append(ind_LF,L_uncer_flex)
        i+=1

    H_uncer_spe_std = np.std(ind_HS, axis=0)
    H_uncer_spe_means = np.mean(ind_HS, axis=0)
    H_uncer_flex_std = np.std(ind_HF, axis=0)
    H_uncer_flex_means = np.mean(ind_HF, axis=0)
    L_uncer_spe_std = np.std(ind_LS, axis=0)
    L_uncer_spe_means = np.mean(ind_LS, axis=0)
    L_uncer_flex_std = np.std(ind_LF, axis=0)
    L_uncer_flex_means = np.mean(ind_LF, axis=0)

    H_uncer_std = [H_uncer_spe_std,H_uncer_flex_std]
    H_uncer_means = [H_uncer_spe_means,H_uncer_flex_means]
    L_uncer_std = [L_uncer_spe_std,L_uncer_flex_std]
    L_uncer_means = [L_uncer_spe_means,L_uncer_flex_means]
    H_uncer_raw = [ind_HS,ind_HF]
    L_uncer_raw = [ind_LS,ind_LF]
    return L_uncer_means,L_uncer_std,H_uncer_means,H_uncer_std,L_uncer_raw,H_uncer_raw


def viz_bar(data_dict,data_name,indic,indicname,y_lim = [0,1],y_ticks = [0,0.5,1]):
    print(f'Proccessing data of {indicname}')
    fig, ax = plt.subplots(1, len(data_dict), figsize=(18, 4))
    fig.text(0.5, 0.1, 'Goal condition', ha='center', va='center')

    for i,data_allsubj in enumerate(data_dict.keys()):
        L_uncer_means,L_uncer_std,H_uncer_means,H_uncer_std,L_uncer_raw,H_uncer_raw = get_bardata(data_dict[data_allsubj],indic)
        x = np.arange(len(L_uncer_means)) 
        # 绘制柱状图
        width = 0.35
        ax[i].bar(x - width/2, L_uncer_means, width, yerr=L_uncer_std, color = [150/255,204/255,203/255], label='Low Uncertainty' , capsize=5)
        ax[i].bar(x + width/2, H_uncer_means, width, yerr=H_uncer_std, color = [231/255,239/255,250/255], label='High Uncertainty', capsize=5)
        for j in range(len(L_uncer_means)):
            # 添加原始数据点的散点图
            x_scatter1 = np.full(len(L_uncer_raw[j]),x[j] - width/2)
            x_scatter2 = np.full(len(L_uncer_raw[j]),x[j] + width/2)
            ax[i].scatter(x_scatter1, L_uncer_raw[j], marker='o', s=15, color=[150/255,204/255,203/255], edgecolor=[87/255,96/255,105/255], linewidths=0.3, alpha=0.4)
            ax[i].scatter(x_scatter2, H_uncer_raw[j], marker='o', s=15, color=[231/255,239/255,250/255], edgecolor=[87/255,96/255,105/255], linewidths=0.3, alpha=0.4)

        # 设置图表标签和标题
        basic_format(ax[i],'',indicname)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(['specific', 'flexible'])
        ax[i].set_ylim(y_lim[0],y_lim[1])
        ax[i].set_yticks(y_ticks)
        ax[i].set_title(data_name[i])
            

    # 显示图表
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', frameon=False)   
    fig.subplots_adjust(bottom=0.2,top=0.8,wspace=0.3)
    plt.show()


if __name__ == '__main__':
    ## STEP 0: COLOR SETTING  
    agent_color = [[126/255,153/255,244/255],
                   [122/255,65/255,113/255],
                   [204/255,126/255,177/255],
                   [163/255,163/255,162/255],
                   [68/255,97/255,123/255],
                   [255/255,85/255,84/255]]
    
    ## STEP 1: LOAD OR SAVE DATA    
    cfg = datap.load_config()
    dir = cfg["data_dir"]
    sim_mode = cfg["sim_mode"]

    data_name = ['MDT_sim','MB_sim','MF_sim','RA_sim','MUD_true','HC_true']
    data_dict = {}

    for name in data_name:
            dataname = f'behavdata_{name}'
            agent_data = datap.load_pkl(dir,dataname)
            data_dict[name] = agent_data

    ## STEP 2: PLOT DATA BY BAR
    indic_viz_param = {}
    indic_viz_param['Mean reward'] = viz_param("datap.rew",[0,40],[0,20,40])
    indic_viz_param['Hit rate'] = viz_param("datap.hr")
    indic_viz_param['Proportion of optimal choices'] = viz_param("datap.poc")
    
    for indicname in indic_viz_param.keys():    
        indic = indic_viz_param[indicname].indic
        y_lim = indic_viz_param[indicname].lim
        y_ticks = indic_viz_param[indicname].ticks
        viz_bar(data_dict,data_name,indic,indicname,y_lim,y_ticks)
    
    ## STEP 3: PLOT DATA BY CURVE
    indic_viz_param['Mean reward']= viz_param("datap.rew",[0,3000],[0,1500,3000])
    indic_viz_param['Hit rate'] = viz_param("datap.hr")
    indic_viz_param['Proportion of optimal choices'] = viz_param("datap.poc",[0,1],[0,0.5,1])

    for indicname in indic_viz_param.keys():    
        indic = indic_viz_param[indicname].indic
        y_lim = indic_viz_param[indicname].lim
        y_ticks = indic_viz_param[indicname].ticks
        viz_curve(data_dict,data_name,agent_color,indic,indicname,y_lim,y_ticks)



