import pickle

# 1. 读取原始pkl文件
with open('/home/data/yufengspace/25MDT/fitdata/fitresults_Hybrid_MUD.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# 2. 定义需要剔除的被试
exclude_subjects = ['Q20572','Q20742','Q20788','Q20895','Q21082','Q21166','Q21177','Q21184']  # 替换为你要剔除的被试ID

# 3. 删除指定被试
for subj in exclude_subjects:
    if subj in data_dict:
        del data_dict[subj]
        print(f"已删除被试: {subj}")
    else:
        print(f"警告: 被试 {subj} 不存在于数据中")

# 4. 保存修改后的数据
with open('/home/data/yufengspace/25MDT/fitdata/fitresults_Hybrid_MUD.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print(f"\n剔除完成！剩余被试数量: {len(data_dict)}")