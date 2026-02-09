import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取数据 ====================
print("="*60)
print("通道Beta值与行为参数相关分析")
print("="*60)

# 读取两个summarydata文件
earnings_data = pd.read_excel('D:/CCNN/Projects/MDT/Data/summary_Earnings.xlsx')
mbprob_data = pd.read_excel('D:/CCNN/Projects/MDT/Data/summary_MBProb.xlsx')

print(f"\nEarnings数据shape: {earnings_data.shape}")
print(f"MBProb数据shape: {mbprob_data.shape}")

# ==================== 2. 定义分析函数 ====================
def analyze_correlations(data, condition_name, target_vars=['MB prob', 'Optimal choice rate']):
    """
    分析通道beta值与行为参数的相关关系
    
    Parameters:
    - data: summary数据框
    - condition_name: 条件名称（用于报告）
    - target_vars: 要分析的行为参数列表
    """
    print(f"\n{'='*60}")
    print(f"分析条件: {condition_name}")
    print(f"{'='*60}")
    
    # 2.1 识别通道beta列（以ch开头，_beta结尾的列）
    channel_cols = [col for col in data.columns if col.startswith('ch') and col.endswith('_beta')]
    print(f"找到 {len(channel_cols)} 个通道beta列")
    print(f"通道列: {channel_cols}")
    
    if len(channel_cols) == 0:
        print("⚠️ 警告: 没有找到通道beta列！")
        return None
    
    # 2.2 检查目标变量是否存在
    available_targets = [var for var in target_vars if var in data.columns]
    if len(available_targets) == 0:
        print(f"⚠️ 警告: 目标变量 {target_vars} 都不存在于数据中！")
        print(f"可用的列: {data.columns.tolist()}")
        return None
    
    print(f"分析的行为参数: {available_targets}")
    
    # 2.3 存储结果
    results = []
    
    # 2.4 对每个目标变量和每个通道进行相关分析
    for target_var in available_targets:
        print(f"\n分析 {target_var} 与各通道的相关:")
        
        for channel_col in channel_cols:
            # 提取通道编号
            channel_num = channel_col.replace('ch', '').replace('_beta', '')
            
            # 删除缺失值
            valid_data = data[[channel_col, target_var]].dropna()
            
            if len(valid_data) < 3:
                print(f"  通道 {channel_num}: 数据不足 (n={len(valid_data)})")
                continue
            
            # 计算Pearson相关
            r, p = stats.pearsonr(valid_data[channel_col], valid_data[target_var])
            
            # 存储结果
            results.append({
                'condition': condition_name,
                'target_variable': target_var,
                'channel': channel_num,
                'channel_col': channel_col,
                'r': r,
                'p_value': p,
                'n': len(valid_data),
                'significant': p < 0.05
            })
    
    # 2.5 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("⚠️ 没有有效的相关分析结果")
        return None
    
    # 2.6 多重比较校正（Bonferroni校正）
    n_tests = len(results_df)
    results_df['p_bonferroni'] = results_df['p_value'] * n_tests
    results_df['p_bonferroni'] = results_df['p_bonferroni'].clip(upper=1.0)
    results_df['significant_bonferroni'] = results_df['p_bonferroni'] < 0.05
    
    # 2.7 按p值排序
    results_df = results_df.sort_values('p_value')
    
    return results_df, data, channel_cols, available_targets

# ==================== 3. 分析两个条件 ====================
earnings_results, earnings_full, earnings_channels, earnings_targets = analyze_correlations(
    earnings_data, 'Earnings',target_vars = ['No-plan impulsivity', 'Optimal choice rate']
)

mbprob_results, mbprob_full, mbprob_channels, mbprob_targets = analyze_correlations(
    mbprob_data, 'MBProb',target_vars = ['No-plan impulsivity', 'Optimal choice rate']
)

# ==================== 4. 生成详细报告 ====================
def print_results(results_df, condition_name):
    """打印分析结果"""
    if results_df is None or len(results_df) == 0:
        return
    
    print(f"\n{'='*60}")
    print(f"{condition_name} - 相关分析结果汇总")
    print(f"{'='*60}")
    
    for target_var in results_df['target_variable'].unique():
        print(f"\n目标变量: {target_var}")
        print("-"*60)
        
        target_results = results_df[results_df['target_variable'] == target_var]
        
        # 显著相关（未校正）
        sig_results = target_results[target_results['significant']]
        print(f"\n显著相关的通道 (p < 0.05, 未校正): {len(sig_results)} 个")
        if len(sig_results) > 0:
            for _, row in sig_results.iterrows():
                print(f"  通道 {row['channel']}: r={row['r']:.3f}, p={row['p_value']:.4f}, n={row['n']}")
        
        # 显著相关（Bonferroni校正后）
        sig_bonf = target_results[target_results['significant_bonferroni']]
        print(f"\n显著相关的通道 (p < 0.05, Bonferroni校正): {len(sig_bonf)} 个")
        if len(sig_bonf) > 0:
            for _, row in sig_bonf.iterrows():
                print(f"  通道 {row['channel']}: r={row['r']:.3f}, p={row['p_value']:.4f}, p_corr={row['p_bonferroni']:.4f}, n={row['n']}")

# 打印结果
print_results(earnings_results, 'Earnings')
print_results(mbprob_results, 'MBProb')