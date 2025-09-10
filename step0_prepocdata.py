import os
import pandas as pd
import datap



if __name__ == "__main__":

    cfg = datap.load_config()
    dir = cfg["data_dir"]

    behav_path = os.path.join(dir, "preprocdata", "behavior")
    MUD_pkl = os.path.join(dir, "preprocdata", "MUD_alldata.pkl")
    HC_pkl = os.path.join(dir, "preprocdata", "HC_alldata.pkl")

    # 用于存储所有被试数据
    MUD_alldata = {}
    HC_alldata = {}

    # 遍历文件夹下的所有xlsx文件
    for file in os.listdir(behav_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(behav_path,file)
            df = pd.read_excel(file_path)
            
            # 去掉扩展名
            filename_no_ext = os.path.splitext(file)[0]
            
            # 分组逻辑
            if filename_no_ext.startswith("Q"):
                # 病人组
                subject_id = filename_no_ext.split("_")[0]  # 提取 ID 部分
                MUD_alldata[subject_id] = df
            elif filename_no_ext[:3].isdigit():
                # 健康组
                subject_id = filename_no_ext.split("_")[0]
                HC_alldata[subject_id] = df

    # 保存为pkl文件
    pd.to_pickle(MUD_alldata, MUD_pkl)
    pd.to_pickle(HC_alldata, HC_pkl)

    print(f"MUD组: {len(MUD_alldata)} 个被试，已保存到 {MUD_pkl}")
    print(f"HC组: {len(HC_alldata)} 个被试，已保存到 {HC_pkl}")