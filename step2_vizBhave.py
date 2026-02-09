from matplotlib import font_manager
from package import datap
from scipy.ndimage import gaussian_filter1d
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'regular'

class viz_param:
    def __init__(self,indic,lim = [0,1],ticks = [0,0.5,1]):
        self.indic = eval(indic)
        self.lim = lim
        self.ticks = ticks

def make_wide(df, varname):
    sub = df[['Subjnum', 'Group label', 'Stage label', varname]]
    wide = sub.pivot_table(index=['Subjnum','Group label'],
                        columns='Stage label',
                        values=varname).reset_index()
    wide.columns.name = None
    return wide.rename(columns={'pre':f'{varname} pre',
                                'post':f'{varname} post',
                                'follow-up':f'{varname} follow-up'})

def basic_format(ax, x_name, y_name, legend='off', linewidth=1.5,
                multiplots=None, ratio=None, title=None, fontsize=14):
    
    if legend == 'on':
        ax.legend(loc='lower left', bbox_to_anchor=(0.7, 1), frameon=False, fontsize = fontsize-4)
    elif legend == 'off' and ax.get_legend() is not None:
        ax.get_legend().remove()

    if multiplots == None:
        ax.set_position([0.2, 0.2, 0.6, 0.6])

    if title != None:
        ax.text(
        0.5, 1.2, title,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment='center',
        horizontalalignment='center',
        bbox=None
    )
        
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.tick_params(axis='both', which='major', width=linewidth, length=7)
    ax.tick_params(axis='both', which='minor', width=linewidth, length=7)

    # 去除边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel(y_name, fontsize=fontsize)
    ax.set_xlabel(x_name, fontsize=fontsize)
    ax.set_box_aspect(ratio)        
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_axisbelow(False)


def violin(ax, data, x, y, order = None, palette = None, orient='v',
        hue=None, hue_order=None, 
        mean_marker_size=6, err_capsize=.11, scatter_size=5):

        if hue is not None and hue_order is None:
            hue_order = sorted(data[hue].dropna().unique())

        g_var = y if orient=='h' else x
        v_var = x if orient=='h' else y
        v=sns.violinplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=hue, hue_order=hue_order,
                            orient=orient, palette=palette, 
                            alpha=.05, inner=None, density_norm='width',
                            legend=False, clip_on=True,
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
                            zorder=2, clip_on=False,
                            ax=ax)
        sns.barplot(data=data, 
                            x=x, y=y, order=order, 
                            orient=orient, 
                            hue=hue, hue_order=hue_order,
                            errorbar='sd', linewidth=2, 
                            edgecolor=(0,0,0,0), facecolor=(0,0,0,0),
                            capsize=err_capsize, legend=False,
                            ax=ax)

        groupby = [g_var, hue] if hue is not None else [g_var]
        sns.stripplot(data=data.groupby(by=groupby)[v_var].mean().reset_index(), 
                        x=x, y=y, order=order, 
                        hue=hue, hue_order=hue_order, 
                        palette=[[.2]*3]*len(hue_order) if hue is not None else None,
                        dodge=False if hue is None else True, 
        
                        marker='o', size=mean_marker_size, color=[.2]*3, legend=False, ax=ax)
        
def viz_curve(ax, data, x, y , smooth = None, hue = None, hue_order=None, orient='x',
              palette = None):
    
    '''
    smooth:
        None: 不平滑
        True: 平滑
    '''
    data_plot = data.copy()
    y_col = y

    if hue is not None and hue_order is None:
        hue_order = sorted(data_plot[hue].dropna().unique())

    if smooth:
        # 创建一个新列存平滑后的值
        y_smooth = y + '_smooth'
        data_plot[y_smooth] = data_plot[y].values  # 先复制一份
        if hue is None:
            data_plot = data_plot.sort_values(by=x)
            data_plot[y_smooth] = gaussian_filter1d(data_plot[y_smooth].values, sigma=2)
        else:
            smoothed = []
            for g in hue_order:
                sub = data_plot[data_plot[hue] == g].copy()
                sub = sub.sort_values(by=x)
                sub[y_smooth] = gaussian_filter1d(sub[y].values, sigma=2)
                smoothed.append(sub)
            data_plot = pd.concat(smoothed, ignore_index=True)
        y_col = y_smooth

    sns.lineplot(
        data=data_plot, x=x, y=y_col,
        hue=hue, hue_order = hue_order, orient=orient,
        estimator="mean", errorbar=('ci', 95), err_kws={'alpha': 0.1},
        palette=palette, linewidth=3,
        ax = ax)


if __name__ == '__main__':
    ## STEP 0: COLOR SETTING  
    group_color = [[184/255,96/255,120/255],
                   [174/255,214/255,213/255]]
    
    ## STEP 1: LOAD OR SAVE DATA    
    cfg = datap.load_config()
    dir = cfg["data_dir"]
    trldata_name = 'alldata_trlbytrl.xlsx'
    summarydata_name = 'alldata_condisummary.xlsx'

    trlbytrldata = pd.read_excel(f'{dir}/{trldata_name}') 
    summarydata = pd.read_excel(f'{dir}/{summarydata_name}') 

    Spe_summarydata = summarydata[summarydata['Goal']=='Specific']
    Flex_summarydata = summarydata[summarydata['Goal']=='Flexible']
    

    ## STEP 2: PLOT DATA BY VIOLIN
    x = 'Uncertainty'
    y = 'Optimal choice rate'
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.2), dpi=150)
    violin(axs[0], Spe_summarydata, x=x, y=y, order=['Low','High'],
           hue='Group label', hue_order=['MUD','HC'], palette=group_color)
    axs[0].set_ylim([0, 1])
    axs[0].set_yticks([0, 0.5, 1])
    axs[0].set_position([0.12, 0.2, 0.35, 0.6])
    basic_format(axs[0], '', y, legend='off', multiplots = True, ratio=1.1,
                title='Specific', fontsize=12)

    violin(axs[1], Flex_summarydata, x=x, y=y, order=['Low','High'],
           hue='Group label', hue_order=['MUD','HC'], palette=group_color)
    axs[1].set_ylim([0, 1])
    axs[1].set_yticks([0, 0.5, 1])
    axs[1].set_position([0.6, 0.2, 0.35, 0.6])
    basic_format(axs[1], '', y, legend='on', multiplots = True, ratio=1.1,
                title='Flexible', fontsize=12)
    fig.savefig(f'D:\CCNN\Projects\MDT\Results\Truebehav_{x}_{y}.SVG', dpi=300)

    # STEP 3: PLOT DATA TRIAL BY TRIAL
    x = 'Trials'
    y = 'Optimal choice rate'
    fig, ax = plt.subplots(figsize=(4.4, 2.2), dpi=150)
    ax.text(0.125, 0.25, "LS", transform=ax.transAxes, zorder=0,
            color='black', alpha=0.5, ha='center', va='center', fontsize=12)
    ax.text(0.375, 0.25, "LF", transform=ax.transAxes, zorder=0,
            color='black', alpha=0.5, ha='center', va='center', fontsize=12)
    ax.text(0.625, 0.25, "HS", transform=ax.transAxes, zorder=0,
            color='black', alpha=0.5, ha='center', va='center', fontsize=12)
    ax.text(0.875, 0.25, "HF", transform=ax.transAxes, zorder=0,
            color='black', alpha=0.5, ha='center', va='center', fontsize=12)
    viz_curve(ax, trlbytrldata, x=x, y=y,
            hue='Group label', hue_order=['MUD','HC'], palette = group_color)
    ax.set_xlim([1,150])
    ax.set_xticks([1, 37, 75, 112, 150])
    ax.set_ylim([0.6, 1])
    ax.set_yticks([0.6, 0.8, 1])
    basic_format(ax, x, y, legend='on', fontsize=12)
    ax.set_position([0.2, 0.25, 0.6, 0.6])
    ax.legend(loc='lower left', bbox_to_anchor=(0.9, 0.9), frameon=False, fontsize = 8)
    fig.savefig(f'D:\CCNN\Projects\MDT\Results\Truebehav_{x}_{y}.SVG', dpi=300)
    plt.show()


