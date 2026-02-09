import pandas as pd
import numpy as np

# ==================== 1. 读取数据 ====================
# 读取MUD组长数据
mud_data = pd.read_csv('D:/CCNN/Projects/MDT/xyf/alldata_summary/preprocdata/fNIRS/MUD/betas.csv')
print(f"MUD组数据shape: {mud_data.shape}")
print(f"MUD组列名: {mud_data.columns.tolist()}")

# 读取HC组长数据
hc_data = pd.read_csv('D:/CCNN/Projects/MDT/xyf/alldata_summary/preprocdata/fNIRS/HC/betas.csv')
print(f"HC组数据shape: {hc_data.shape}")
print(f"HC组列名: {hc_data.columns.tolist()}")

# 读取统计分析结果（包含t值和p值）
stats_data = pd.read_csv('D:/CCNN/Projects/MDT/xyf/alldata_summary/preprocdata/fNIRS/group_all_ttest_fdr.csv')
print(f"\n统计数据shape: {stats_data.shape}")
print(f"统计数据列名: {stats_data.columns.tolist()}")
print(stats_data.head())

# 读取summarydata短数据
summary_data = pd.read_excel('D:/CCNN/Projects/MDT/Data/allfitdata_summary.xlsx')
print(f"\nSummarydata shape: {summary_data.shape}")
print(f"Summarydata列名: {summary_data.columns.tolist()}")

# ==================== 2. 拼接两组长数据 ====================
# 假设长数据包含列: subj, channel, Earnings, MBProb
# 如果需要添加组别标识
if 'group' not in mud_data.columns:
    mud_data['group'] = 'MUD'
if 'group' not in hc_data.columns:
    hc_data['group'] = 'HC'

# ==================== 2.5 修复HC组subj格式问题 ====================
# HC组的subj需要补充前导0，使其变成三位数格式（如 21 -> 021）
# 检查subj列的数据类型，如果是数字则转换为字符串并补0
if hc_data['subj'].dtype in ['int64', 'int32', 'float64', 'float32']:
    hc_data['subj'] = hc_data['subj'].astype(int).astype(str).str.zfill(3)
    print(f"HC组subj已格式化为三位数: {hc_data['subj'].unique()[:5]}")
else:
    # 如果已经是字符串，也尝试补0
    hc_data['subj'] = hc_data['subj'].astype(str).str.zfill(3)
    print(f"HC组subj已格式化为三位数: {hc_data['subj'].unique()[:5]}")

# 同样处理MUD组，确保格式一致（如果MUD组也需要）
# 如果MUD组的格式不同，可以注释掉下面这段
if mud_data['subj'].dtype in ['int64', 'int32', 'float64', 'float32']:
    # 检查MUD组是否也需要格式化
    # 如果MUD组是字母+数字格式，则不需要处理
    try:
        mud_data['subj'] = mud_data['subj'].astype(str)
        print(f"MUD组subj格式: {mud_data['subj'].unique()[:5]}")
    except:
        pass

# 拼接数据
long_data = pd.concat([mud_data, hc_data], ignore_index=True)
print(f"\n拼接后长数据shape: {long_data.shape}")
print(f"拼接后长数据前几行:")
print(long_data.head())

# ==================== 3. 找出显著激活的通道 ====================
# 设置显著性阈值（例如 p < 0.05）
p_threshold = 0.05

# 假设stats_data包含列: channel, condition, t_value, p_value
# 分别为两个条件找出显著通道
significant_channels_earnings = stats_data[
    (stats_data['Param'] == 'Earnings') & 
    (stats_data['p_fdr'] < p_threshold)
]['Channel'].tolist()

significant_channels_mbprob = stats_data[
    (stats_data['Param'] == 'MBProb') & 
    (stats_data['p_fdr'] < p_threshold)
]['Channel'].tolist()

print(f"\nEarnings显著通道数量: {len(significant_channels_earnings)}")
print(f"Earnings显著通道编号: {significant_channels_earnings}")
print(f"\nMB prob显著通道数量: {len(significant_channels_mbprob)}")
print(f"MB prob显著通道编号: {significant_channels_mbprob}")

# ==================== 4. 处理函数：为每个条件生成summary文件 ====================
def process_condition(long_data, summary_data, significant_channels, beta_column, condition_name):
    """
    为指定条件处理数据并生成summary文件
    
    Parameters:
    - long_data: 拼接后的长数据
    - summary_data: 原始summarydata
    - significant_channels: 该条件的显著通道列表
    - beta_column: beta值所在的列名（'Earnings' 或 'MBProb'）
    - condition_name: 条件名称（用于文件命名）
    """
    print(f"\n{'='*60}")
    print(f"处理条件: {condition_name} (列名: {beta_column})")
    print(f"{'='*60}")
    
    # 4.1 筛选只保留显著通道的数据
    condition_data = long_data[
        long_data['channel'].isin(significant_channels)
    ][['subj', 'channel', beta_column]].copy()
    
    print(f"筛选后数据shape: {condition_data.shape}")
    
    if len(condition_data) == 0:
        print(f"⚠️ 警告: {condition_name}条件没有找到数据！")
        return None
    
    # 4.2 转换为宽数据格式
    # 每个被试一行，每个显著通道一列
    wide_data = condition_data.pivot(
        index='subj', 
        columns='channel', 
        values=beta_column
    ).reset_index()
    
    # 重命名列，添加前缀以便识别
    wide_data.columns = ['subj'] + [f'ch{col}_beta' for col in wide_data.columns[1:]]
    print(f"转换为宽数据后shape: {wide_data.shape}")
    print(f"宽数据列名: {wide_data.columns.tolist()}")
    
    # 4.3 获取做过通道分析的被试ID列表
    analyzed_subjects = wide_data['subj'].tolist()
    
    # 4.4 筛选summarydata，只保留做过通道分析的被试
    summary_filtered = summary_data[summary_data['Subjnum'].isin(analyzed_subjects)].copy()
    print(f"剔除前summarydata被试数: {len(summary_data)}")
    print(f"剔除后summarydata被试数: {len(summary_filtered)}")
    
    # 4.5 合并数据（注意两个数据框的被试列名不同）
    final_data = summary_filtered.merge(
        wide_data, 
        left_on='Subjnum',
        right_on='subj', 
        how='left'
    )
    
    # 删除重复的subj列，保留Subjnum
    if 'subj' in final_data.columns:
        final_data = final_data.drop('subj', axis=1)
    print(f"最终数据shape: {final_data.shape}")
    print(f"最终数据列数: {len(final_data.columns)}")
    
    # 4.6 保存结果
    output_filename = f'summary_{condition_name}.xlsx'
    final_data.to_excel(output_filename, index=False, encoding='utf-8-sig')
    print(f"✓ 已保存至: {output_filename}")
    
    # 4.7 显示预览
    print(f"\n{condition_name}数据预览:")
    print(final_data.head())
    
    return final_data

# ==================== 5. 分别处理两个条件 ====================
earnings_summary = process_condition(
    long_data, 
    summary_data, 
    significant_channels_earnings, 
    'Earnings',
    'Earnings'
)

mbprob_summary = process_condition(
    long_data, 
    summary_data, 
    significant_channels_mbprob, 
    'MBProb',
    'MBProb'
)

# ==================== 6. 生成总体处理报告 ====================
print("\n" + "="*60)
print("总体处理报告")
print("="*60)
print(f"原始MUD组被试数: {len(mud_data['subj'].unique())}")
print(f"原始HC组被试数: {len(hc_data['subj'].unique())}")
print(f"总通道数: 40")
print(f"显著性阈值: p < {p_threshold}")
print(f"\n条件1: Earnings")
print(f"  - 显著通道数: {len(significant_channels_earnings)}")
print(f"  - 最终被试数: {len(earnings_summary) if earnings_summary is not None else 0}")
print(f"\n条件2: MBProb")
print(f"  - 显著通道数: {len(significant_channels_mbprob)}")
print(f"  - 最终被试数: {len(mbprob_summary) if mbprob_summary is not None else 0}")
print("="*60)
print("\n✓ 所有处理完成！")
print("生成的文件:")
print("  1. summary_Earnings.xlsx")
print("  2. summary_MBProb.xlsx")