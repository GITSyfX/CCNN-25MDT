from matplotlib import font_manager
from package import datap
from scipy.ndimage import gaussian_filter1d
import numpy as np  
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'regular'
plt.rcParams['svg.fonttype'] = 'none'

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
        
def viz_corr(ax, data, x, y, color=None):
    sns.regplot(
    data=data, x=x, y=y,
    scatter_kws={"s": 40, "alpha": 0.7, "clip_on": False},
    line_kws={"linewidth": 3}, ci=95, color=color,
    ax=ax)

    df = data[[x, y]].dropna()
    r, p = stats.pearsonr(df[x], df[y])

    textstr = f"$r$ = {r:.3f}\n$p$ = {p:.3f}"

    # 在右上角添加文字（坐标用轴的 fraction）
    ax.text(
        0.61, 1, textstr,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=None
    )


if __name__ == '__main__':
    ## STEP 0: COLOR SETTING  
    group_color = [[169/255, 169/255,169/255],
                    [184/255,96/255,120/255],
                    [174/255,214/255,213/255]]
    
    ## STEP 1: LOAD OR SAVE DATA    
    cfg = datap.load_config()
    dir = cfg["data_dir"]
    summarydata_name = 'summary_MBprob.xlsx'
    summarydata = pd.read_excel(f'{dir}/{summarydata_name}') 

    condisummarydata_name = 'alldata_condisummary.xlsx'
    condisummarydata = pd.read_excel(f'{dir}/{condisummarydata_name}') 
    
    ## STEP 2: PLOT DATA BY VIOLIN
    # x = 'Group label'
    # y = 'Hit rate'
    # fig, ax = plt.subplots(figsize=(4.4, 2.2), dpi=150)
    # violin(ax, summarydata, x=x, y=y, order=['MUD','HC'],
    #         palette=[group_color[1], group_color[2]])
    # ax.set_ylim([0.4, 1])
    # ax.set_yticks([0.4, 0.7, 1])
    # ax.set_position([0.12, 0.2, 0.35, 0.6])
    # basic_format(ax, '', y, legend='off', ratio=1.1, fontsize=12)

    #fig.savefig(f'D:\CCNN\Projects\MDT\Results\Truebehav_{x}_{y}.SVG', dpi=300)
    #plt.show()

    #STEP 3: PLOT DATA OF CORRELATION

    x = 'No-plan impulsivity'
    y = 'ch3_beta'

    group = ['All','MUD','HC']
    MUD_summarydata = summarydata[summarydata['Group label'] == 'MUD']
    HC_summarydata = summarydata[summarydata['Group label'] == 'HC']


    fig, ax = plt.subplots(figsize=(4.4, 2.2), dpi=150)
    viz_corr(ax, MUD_summarydata , x=x, y=y, color=group_color[1])
    basic_format(ax, x, y, legend='on', ratio=1.1, fontsize=12)
    # ax.set_xlim([0, 1])
    # ax.set_xticks([0, 0.5, 1])
    # ax.set_ylim([0.6, 1])
    # ax.set_yticks([0.6, 0.8, 1])
    ax.set_position([0.2, 0.25, 0.6, 0.6])
    plt.grid(False) 
    fig.savefig(f'D:\CCNN\Projects\MDT\Results\Truecli_{x}_{y}_{group[1]}.SVG', dpi=300, format='svg', bbox_inches='tight')
    plt.show()


