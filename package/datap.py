import yaml
import pickle
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


'''preprocess data '''
def rep_choice(value):
    if value[-1] == 'L':
        return '0'
    else:
        return '1'
 
def preproc(data):
    '''Preprocess the data
    '''
    stage_name = 'stage'
    stage_value = 'train'
    s0_name = 's0'
    s0_value = 0
    col_dict = {
        'this_uncer': 'p', # transitional p seed; not the param p of rng.choice 
        'this_coin_inx': 'g', #goal
        'subres1': 'a1',
        'state_mid': 's1',
        'subres2': 'a2',
        'state_R': 's2',
        'this_reward': 'r2',
    }

    # rename  
    data.rename(columns=col_dict, inplace=True)
    vi = ['g', 'p', 'a1' , 's1', 'a2', 's2', 'r2']
    data = data[vi].copy()

    data['a1'] = data['a1'].replace({'choiceL': '0', 'choiceR': '1'})
    data['a2'] = data['a2'].apply(rep_choice)

    data[vi] = data[vi].apply(pd.to_numeric)

    data['g'] = data['g'].replace(4,-1)
    data['g'] = data['g'].replace(3,6)
    data['g'] = data['g'].replace(2,7)

    rules = np.array([
    [1, 2, 3, 4],
    [6, 7, 7, 8],
    [7, 8, 6, 8],
    [6, 5, 5, 8],
    [6, 8, 8, 5]])  

    for i in range(1, data['s1'].max()):
        idx = data['s1'] == i
        for old_val in [1, 2, 3, 4]:
            new_val = rules[i, old_val-1]  
            mask = idx & (data['s2'] == old_val)
            data.loc[mask, 's2'] = new_val
        
    data.insert(2, s0_name, s0_value)
    data.insert(3, stage_name, stage_value)
    return data 

''' load and save'''
def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file {config_path} not found. Please create one from config.example.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_dir(dir_path):
    "load data to dict forms"
    print('loading all data...')
    alldata = {}
    try:
        subjlist = sorted(os.listdir(dir_path))  
    except Exception as e:
        print(f'Unable to list {dir_path},error: {e}')
        return alldata
    
    for dataname in tqdm(subjlist): 
        filepath = os.path.join(dir_path, dataname)
        if not os.path.isfile(filepath):
            # skip non-files(folder,etc.)
            continue
        try:
            data = pd.read_excel(filepath)
            alldata[dataname] = data
        except Exception as e:
            print(f'Warning: fail to load file {dataname}, error: {e}')
    return alldata


def load_pkl(dir,dataname):
    "load pkl if have parmas"
    with open(f'{dir}/{dataname}.pkl', 'rb') as f:
        agent_data = pickle.load(f)

    for subjname in agent_data.keys():
        data = agent_data[subjname]
        if isinstance(data, list) and len(data) == 2 and isinstance(data[1], pd.DataFrame):
            data = data[1]
        elif isinstance(data, list) and len(data) > 1:
            data = data[-1]
        agent_data[subjname] = data
    
    return agent_data

def load_data(agent,fit_alldata,sim_alldata = []):
    p_trans = agent.p_trans

    rows = []
    for subj_id, results in fit_alldata.items():
        if 'Q' in subj_id:
            group = 'MUD'
        else:
            group = 'HC'

        row = {
            'group': group,
            'subj_id': subj_id,
            'log_post': results['log_post'],
            'negloglike': results['negloglike'],
            'aic': results['aic'],
            'bic': results['bic'],
            'n_param': results['n_param']
        }
        # add fit params
        params = results['param']
        params = [fn(p) for p, fn in zip(params, p_trans)]

        for i, param_name in enumerate(results['param_name']):
            row[param_name] = params[i]


        if sim_alldata: # add raw params for simulation
            sim_list = sim_alldata[subj_id] #sim_list = [rparam,simdata]
            rparams = sim_list[0]
            rparams = [fn(p) for p, fn in zip(rparams, p_trans)]
            for i, param_name in enumerate(results['param_name']): 
                rparam_name = 'r_' + param_name
                row[rparam_name] = rparams[i]

        rows.append(row)

    return pd.DataFrame(rows)

def save(dir):
    subjlist = os.listdir(dir)
    for dataname in subjlist:
        rawdata = pd.read_excel(f'{dir}/{dataname}')
        data = preproc(rawdata)
        data.to_excel(f'Pre-processed_{dataname}',index=False)

''' simulate data'''
def datapush(env, subj,row, rng,flag,last_g):
    # ---------- Stage 1 ----------- #
    # see state 
    g = row['g']
    p = row['p']
    s0  = row['s0'] #s0 = env.reset()
    
    if flag == 1:
        subj.bw_update(g)
        if subj.name == 'MDT' and last_g == -1:
            subj.ind_active_model = 2 # switching the mode
            subj.MB_prob_prev = 0.2 #changing the choice prob accordingly
            subj.MB_prob = subj.MB_prob_prev
        elif subj.name == 'MDT' and last_g != -1:
            subj.ind_active_model = 1 
            subj.MB_prob_prev = 0.8 
            subj.MB_prob = subj.MB_prob_prev
            
    # the next state, rew, and done 
    pi1  = subj.policy(s0)
    a1  = rng.choice(env.nA, p=pi1)
    s1,r1,done = env.step(s0,a1,g,p)

    pi2 = subj.policy(s1)
    a2 = rng.choice(env.nA, p=pi2)
    #save the info 
    subj.mem.push({
        'g': g, 
        'p': p,
        's': s0, 
        's_next': s1,
        'a': a1, 
        'a_next': a2,     
        'r': r1,
        'pi1': pi1, 
        'done': done
    })
    subj.learn()

    # ---------- Stage 2 ----------- #

    s2,r2,done = env.step(s1,a2,g,p)
    #save the info 
    subj.mem.push({
        'g': g, 
        'p': p,
        's': s1, 
        's_next': s2,
        'a': a2, 
        'a_next': 'nochoice',
        'r': r2,
        'pi2': pi2,
        'done': done
    })
    subj.learn()
    return a1, s1, pi1, a2, s2, pi2, r2

def block(agent,env,init,seed): 
    rng = np.random.RandomState(seed)
    if init:
        # if there are assigned params
        params = init   
    else:
        # random init from the possible bounds 
        pbnds = [[fn(p) for p in pbnd] for fn, pbnd in 
                zip(agent.p_links, agent.pbnds)]
        
        params = [pbnd[0] + (pbnd[1] - pbnd[0]
                    ) * rng.rand() for pbnd in pbnds]
        
    n_rep1=37
    n_rep2=38
    data = {}
    n_rep = n_rep1*2+n_rep2*2 
    block_g = [6]*n_rep1 + [-1]*n_rep2 + [7]*n_rep1 + [-1]*n_rep2 
    block_p = [0.9]*n_rep1 + [0.9]*n_rep2 + [0.5]*n_rep1 + [0.5]*n_rep2
    # block_g = [7]*n_rep1 +  [-1]*n_rep2 + [5]*n_rep1 + [-1]*n_rep2 
    # block_p = [0.9]*n_rep1 + [0.9]*n_rep2 + [0.5]*n_rep1 + [0.5]*n_rep2
    data['g']  = block_g
    data['p']  = block_p
    data['s0']     = [0]*n_rep
    data['stage']  = ['learning']*n_rep

    block_data = pd.DataFrame.from_dict(data)

    ## init the agent 
    subj = agent(env,params)

    ## init a blank dataframe to store simulation
    col = ['a1', 's1', 'pi1', 'a2', 's2', 'pi2','r2']
    init_mat = np.zeros([block_data.shape[0], len(col)]) 
    pred_data = pd.DataFrame(init_mat, columns=col)  

    ## loop to simulate the responses in the block
    last_g = -1 
    for t, row in block_data.iterrows():
        if row['g'] != last_g:
            flag = 1
            last_g = row['g']
        else:
            flag = 0
        # simulate the data 

        a1, s1, pi1, a2, s2, pi2, r2 = datapush(env,subj,row,rng,flag,last_g)
        
        # record the stimulated data 
        for c in col: 
            if c == 'pi1' or c == 'pi2':
                pred_data[c] = pred_data[c].astype('object')
                pred_data.at[t, c] = eval(c)
            else:
                pred_data.loc[t, c] = eval(c)
                pred_data[c] = pred_data[c].astype('int')  

    simdata = pd.concat([block_data, pred_data], axis=1)
    return params,simdata


''' evaluate decision'''
def rew(data):
    reward = []
    rev = 0
    for i in range(len(data)):
        r = data.loc[i,'r2']
        rev += r
        reward.append(rev)
    reward = np.array(reward)
    
    # L_uncer_spe_rew = (reward[112]-reward[0])/len(reward[0:112])
    # L_uncer_flex_rew = (reward[225]-reward[113])/len(reward[113:225])
    # H_uncer_spe_rew = (reward[338]-reward[226])/len(reward[226:338])
    # H_uncer_flex_rew = (reward[451]-reward[339])/len(reward[339:451])
    L_uncer_spe_rew = (reward[36]-reward[0])/len(reward[0:36])
    L_uncer_flex_rew = (reward[74]-reward[37])/len(reward[37:74])
    H_uncer_spe_rew = (reward[111]-reward[75])/len(reward[75:111])
    H_uncer_flex_rew = (reward[149]-reward[112])/len(reward[112:149])

    return reward,H_uncer_spe_rew,H_uncer_flex_rew,L_uncer_spe_rew,L_uncer_flex_rew

def hr(data):
    hit = []
    hr = []
    t = 0
    for i in range(len(data)):
        if data.loc[i,'r2'] > 0:
            t+=1
        hit.append(t)
        hr.append(t/(i+1))
    hit = np.array(hit)
    hr = np.array(hr)
    
    # L_uncer_spe_hr = (hit[112]-hit[0])/len(hit[0:112])
    # L_uncer_flex_hr = (hit[225]-hit[113])/len(hit[113:225])
    # H_uncer_spe_hr = (hit[338]-hit[226])/len(hit[226:338])
    # H_uncer_flex_hr = (hit[451]-hit[339])/len(hit[339:451])
    L_uncer_spe_hr = (hit[36]-hit[0])/len(hit[0:36])
    L_uncer_flex_hr = (hit[74]-hit[37])/len(hit[37:74])
    H_uncer_spe_hr = (hit[111]-hit[75])/len(hit[75:111])
    H_uncer_flex_hr = (hit[149]-hit[112])/len(hit[112:149])

    return hr,H_uncer_spe_hr,H_uncer_flex_hr,L_uncer_spe_hr,L_uncer_flex_hr 

def poc(data):
    poc_rate = []
    poc = []
    t = 0
    for i in range(0,len(data)):
        if data.loc[i,'p'] == 0.9:
            if data.loc[i,'g'] == -1:
                if data.loc[i,'a1'] == 1:
                    t +=1
                if data.loc[i,'s1'] == 1:
                    if data.loc[i,'a2'] == 0:
                        t +=1
                if data.loc[i,'s1'] == 2:
                    if data.loc[i,'a2'] == 1:
                        t +=1
                if data.loc[i,'s1'] == 3:
                    if data.loc[i,'a2'] == 1:
                        t +=1
                if data.loc[i,'s1'] == 4:
                    if data.loc[i,'a2'] == 0:
                        t +=1
            if data.loc[i,'g'] == 7:
                if data.loc[i,'a1'] == 0:
                    t +=1
                if data.loc[i,'s1'] == 1:
                    if data.loc[i,'a2'] == 1:
                        t +=1
                if data.loc[i,'s1'] == 2:
                    if data.loc[i,'a2'] == 0:
                        t +=1
            if data.loc[i,'g'] == 6:
                t += 1
                if data.loc[i,'s1'] == 1:
                    if data.loc[i,'a2'] == 0:
                        t +=1
                if data.loc[i,'s1'] == 2:
                    if data.loc[i,'a2'] == 1:
                        t +=1
                if data.loc[i,'s1'] == 3:
                    if data.loc[i,'a2'] == 0:
                        t +=1
                if data.loc[i,'s1'] == 4:
                    if data.loc[i,'a2'] == 0:
                        t +=1

        if data.loc[i,'p'] == 0.5:
            if data.loc[i,'g'] == -1:
                if data.loc[i,'a1'] == 1:
                    t +=1
                if data.loc[i,'s1'] == 1:
                    if data.loc[i,'a2'] == 0:
                        t +=1
                if data.loc[i,'s1'] == 2:
                    if data.loc[i,'a2'] == 1:
                        t +=1
                if data.loc[i,'s1'] == 3:
                    if data.loc[i,'a2'] == 0:
                        t +=1
                if data.loc[i,'s1'] == 4:
                    if data.loc[i,'a2'] == 1:
                        t +=1
            if data.loc[i,'g'] == 7:
                if data.loc[i,'a1'] == 0:
                    t +=1
                if data.loc[i,'s1'] == 1:
                    t +=1
                if data.loc[i,'s1'] == 2:
                    if data.loc[i,'a2'] == 0:
                        t +=1
            if data.loc[i,'g'] == 6:
                t += 1
                if data.loc[i,'s1'] == 1:
                    if data.loc[i,'a2'] == 0:
                        t +=1
                if data.loc[i,'s1'] == 2:
                    if data.loc[i,'a2'] == 1:
                        t +=1
                if data.loc[i,'s1'] == 3:
                    if data.loc[i,'a2'] == 0:
                        t +=1
                if data.loc[i,'s1'] == 4:
                    if data.loc[i,'a2'] == 0:
                        t +=1 
        poc_rate.append(t/(2*(i+1)))
        poc.append(t)

    poc_rate = np.array(poc_rate)
    poc = np.array(poc)

    # L_uncer_spe_poc = (poc[112]-poc[0])/(2*len(poc[0:112]))
    # L_uncer_flex_poc = (poc[225]-poc[113])/(2*len(poc[113:225]))
    # H_uncer_spe_poc = (poc[338]-poc[226])/(2*len(poc[226:338]))
    # H_uncer_flex_poc = (poc[451]-poc[339])/(2*len(poc[339:451]))
    L_uncer_spe_poc = (poc[36]-poc[0])/(2*len(poc[0:36]))
    L_uncer_flex_poc = (poc[74]-poc[37])/(2*len(poc[37:74]))
    H_uncer_spe_poc = (poc[111]-poc[75])/(2*len(poc[75:111]))
    H_uncer_flex_poc = (poc[149]-poc[112])/(2*len(poc[112:149]))
    
    return poc_rate,H_uncer_spe_poc,H_uncer_flex_poc,L_uncer_spe_poc,L_uncer_flex_poc 


