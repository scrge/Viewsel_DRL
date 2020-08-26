#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Set system parameters: ctrl + f, 
### Run Model
### TRAIN EPISODE ###
### Eval Training set ###
### Eval Testing set ###

import unittest
import pdb, traceback, sys, heapq, time
import random, math, pickle, csv, os.path
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from collections import deque

### BELOW tensorflow 2.0 :
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
# from keras.optimizers import Adam
# from keras.layers.normalization import BatchNormalization
# from keras.callbacks import History 

### tensorflow 2.0 :
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
# from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History 
# import tensorflow.keras.backend as K

# internal modules
from viewsel_env_viewdict import *

class DQN:
    def __init__(self, env, max_mem_chunk_len, save_mem_type):
        self.env     = env
        self.memory  = deque(maxlen=max_mem_chunk_len + 1)
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.01
        self.decay_rate = 0.001
        self.gamma = 0.85
        self.tau = .2
        self.batch_size = 1

        self.model        = self.create_model()
        self.target_model = self.create_model()

        self.qry_rewards_dict = {}
        self.action_plot = []
        self.action_rank_plot = []

        self.history = History()
        self.loss_history = []

        self.save_mem_type = save_mem_type
        self.max_mem_chunk_len = max_mem_chunk_len

    def create_model(self):
        #model.add(Conv1D(num_filters, kernel_size=self.env.RA**2, activation="relu"))
        model   = Sequential()
        state_shape  = self.env.state.shape
        num_filters = self.env.max_num_views
        model.add(Conv1D(8, kernel_size=4, strides=2, padding = 'same',
            input_shape=(state_shape[0], 1 ) ))
        #model.add(BatchNormalization())
        model.add(Activation("relu"))
        #model.add(Dropout(0.5))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(16, kernel_size=4, strides=2, padding = 'same'))
        model.add(Activation("relu"))
        model.add(MaxPooling1D(2))
        # model.add(Conv1D(16, kernel_size=4, strides=2, padding = 'same'))
        # model.add(Activation("relu"))
        # model.add(MaxPooling1D(2))
        model.add(Flatten())
        # model.add(Dense(30, activation='relu'))
        model.add(Dense(self.env.max_num_views))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate, decay=self.decay_rate))
        # model.compile(loss="mean_squared_error",
        #     optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state, *args, **kwargs):  
        file = kwargs.get('file', None)
        rand_var = np.random.random()
        if rand_var > self.epsilon:
            preds = self.model.predict(state)[0][:len(self.env.input_view_strs)]
            #file.write(repr(preds) + '\n')
            inds_of_highest=heapq.nlargest(len(preds), range(len(preds)), key=preds.__getitem__)
            for action in inds_of_highest:
                if action not in self.env.views_tried:
                    break
            if action in self.env.views_tried:
               #file.write('all valid actions for sample chosen\n')
               return -1
            self.env.views_tried.append(action)
            # file.write(repr(inds_of_highest) + '\n')
            # file.write('pos ' + repr(action) + ': ')
            # file.write(repr(self.env.input_view_strs[action]) + '\n')
        else:  #randomly choose new action
            #print('rand')
            k = 0
            while True:
                action = random.randint(0, len(self.env.input_view_strs)-1)
                if (action not in self.env.views_tried):
                    self.env.views_tried.append(action)
                    break
                k += 1
                if k > self.env.max_num_views * 10:
                    return -1 
        return action

    def remember(self, state, action, reward, new_state, done):
        # new_tup = [state, action, reward, new_state, done]
        # already_in = True
        # for tup in self.memory:
        #     for new, old in zip(new_tup, self.memory):
        #         if new != old:
        #             already_in = False
        #             break
        #     if already_in == False:
        #         break
        # if already_in:
        #     self.memory.append([state, action, reward, new_state, done])
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, memory_folder, memory_samp_num, curr_groupno):
        if self.save_mem_type == 'none':
            if len(self.memory) < self.batch_size: 
                return
            samples = random.sample(self.memory, self.batch_size)
        elif self.save_mem_type == 'single':
            if memory_samp_num < self.batch_size:
                return
            samples = []
            nums_used = []
            while len(samples) < self.batch_size:
                sel_samp_num = random.randint(0, memory_samp_num-1)
                if sel_samp_num not in nums_used:
                    nums_used.append(sel_samp_num)
                    sel_samp_fn = str(sel_samp_num)+'.p' 
                    sel_samp = pickle.load(open(memory_folder/sel_samp_fn, 'rb'))    
                    samples.append(sel_samp)
        else:
            if memory_samp_num < self.batch_size:
                return
            samples = []
            nums_used = []
            while len(samples) < self.batch_size:
                sel_samp_num = random.randint(0, memory_samp_num-1) #-1 b/c did +1 after adding latest
                if sel_samp_num not in nums_used:
                    nums_used.append(sel_samp_num)
                    sel_samp_groupno = math.floor(sel_samp_num / self.max_mem_chunk_len)
                    if sel_samp_groupno == 0: 
                        samp_ind = sel_samp_num - 1
                    else:
                        samp_ind = sel_samp_num % self.max_mem_chunk_len
                    if sel_samp_groupno == curr_groupno: #saves time from opening file
                        samples.append(self.memory[samp_ind])
                    else:
                        begin_ind = int( sel_samp_groupno * self.max_mem_chunk_len)
                        end_ind = int( (sel_samp_groupno + 1) * self.max_mem_chunk_len) - 1
                        chunk_fn = str(begin_ind) +'_to_' + str(end_ind) + '.p'
                        mem_chunk = pickle.load(open(memory_folder/chunk_fn, 'rb'))   
                        samples.append(mem_chunk[samp_ind])

        for sample in samples:
            state, action, reward, new_state, done = sample
            # models_preds = self.model.predict(state)
            target = self.target_model.predict(state)
            # for ac in invalids:
            #     target[0][ac] = -1
            Q_future = max(self.target_model.predict(new_state)[0])
            # Q_future = min(max(self.model.predict(new_state)[0]), 
            # max(self.target_model.predict(new_state)[0]))
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
            # self.model.fit(state, target, epochs=1, verbose=0, callbacks=[self.history])
            # self.loss_history.append(self.history.history['loss'][0])  #training loss

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(str(fn))

    def test_act(self, state):
        preds = self.model.predict(state)[0][:len(self.env.input_view_strs)]
        inds_of_highest=heapq.nlargest(len(preds), range(len(preds)), key=preds.__getitem__)
        for action in inds_of_highest:
            if action not in self.env.views_tried:
                break
        self.env.views_tried.append(action)
        return action, preds, inds_of_highest

def record_ep(file, env):
    for v in env.maint_cost_lst.keys():
        file.write(repr(env.input_view_strs[v]) + ', Maint cost: ' + repr(env.maint_cost_lst[v]) +'\n')
    file.write('Rewrites:\n')
    for q in range(env.num_input_qrys):
        file.write('Query ' + repr(q)+ ', ')
        file.write('Proc cost: ' + repr(env.qry_proc_costs[q])+ '\n')
        for i, v in enumerate(env.query_rewrites[q]):  #each view is a list of edges
            if v == 1:
                file.write(repr(env.input_view_strs[i])+ '\n')
    file.write('Default Cost: ' + str(env.init_cost) + '\n')
    file.write('New Cost: ' + repr(env.new_cost) + '\n')
    file.write('reward: ' + repr(env.reward) + '\n')
    file.flush()

    # ep_end = time.time()
    # hours, rem = divmod(ep_end-ep_start, 3600)
    # minutes, seconds = divmod(rem, 60)
    # seconds = round(seconds, 2)
    # print("{:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))
    # file.write(str(hours) + ':' + str(minutes) + ':' + str(seconds) + '\n')   

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

##############################################################################################
### Run Model
##############################################################################################

def main():
    #sys.argv: [1]- run #, [2]- k, [3]-split # : from 1 to k
    #for testing on all new queries: [1]- run #
    ### User variables ###
    # dataset_name = 'Dataset61'
    dataset_name = 'JOB_19'
    trial_name = 'Trial_1'
    num_episodes = 300000
    train_interval = 50000
    test_interval = 10000
    model_save_interval = test_interval
    stop_strike_limit = 15  #if frac_opt > stop thres  for (test_interval * limit) eps, stop
    stop_thres = 0.99
    max_num_queries = 30 #get from pickle in database folder
    max_num_views = 10
    support_frac = 0.25
    max_memory_len = 200000
    max_mem_chunk_len = max_memory_len
    save_mem_type = 'single'  # none, single, chunks ; chunks is too slow if chunksize too large, but fewer # files
    opt_algo_type = 'greedy_newcostmodel'
    opt_algo_type_short = 'greedy'
    eval_trainset = False
    semi_supervised = False
    load_prev_model = False
    write_train_file = False
    write_test_file = False
    num_test_csv_samp_torec = 10  #the no. samples to record. Sum is still over all test_samps. Use value 'all' for all samps
    # num_test_csv_samp_torec = 'all'  #perhaps if > 10, only use first 10
    #######`

    stop_strike = 0
    input_file_dir = Path.cwd().parent / "datasets" / dataset_name 
    vers = 'norewrites_'
    model_name = vers+'run_' +str(sys.argv[1])

    if len(sys.argv) < 3:
        samples = pickle.load(open(input_file_dir/'train_samples.p', 'rb'))    
        test_samples = pickle.load(open(input_file_dir/'test_samples.p', 'rb'))
        suffix = ''
    else:
        all_samples = pickle.load(open(input_file_dir/'train_samples.p', 'rb'))
        k = int(sys.argv[2])
        split_num = int(sys.argv[3]) - 1
        if len(all_samples) % k != 0:
            print('Choose k that evenly divides # of samples')
            sys.exit()
        group_size = len(all_samples) / k
        begin_ind = int( split_num * group_size)
        end_ind = int( (split_num + 1) * group_size)
        suffix = '_k_'+str(k)+'_split_' + str(split_num + 1)

        samples = copy.deepcopy(all_samples)
        del samples[begin_ind : end_ind]    
        test_samples = all_samples[begin_ind : end_ind]
        all_samples = []

    output_folder = dataset_name+', '+trial_name + suffix
    output_path = Path.cwd() / output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    memory_samp_num = 0
    mem_chunk_groupno = 0
    if save_mem_type != 'none':
        fn = 'memory_run_' +str(sys.argv[1])
        memory_folder = Path.cwd() / output_folder / fn
        if not os.path.exists(memory_folder):
            os.makedirs(memory_folder)
    else:
        max_mem_chunk_len = max_memory_len
        memory_folder = None

    opt_fn = 'viewsel_'+opt_algo_type + suffix +'.csv'
    file = open(input_file_dir/opt_fn, 'r')
    opt_test_rewards = []
    begin_record_rewards = False
    read_file = csv.reader(file)
    for line in read_file:
        if line:
            if begin_record_rewards and line[0] != 'Sum':
                opt_test_rewards.append(float(line[-1]))
            if line[-1] == 'Reward':
                begin_record_rewards = True
            end_line = line
    optimal_sum = float(end_line[-1])

    if eval_trainset:
        opt_fn = 'viewsel_train_'+opt_algo_type + suffix+'.csv'
        file = open(input_file_dir/opt_fn, 'r')
        opt_train_rewards = []
        begin_record_rewards = False
        read_file = csv.reader(file)
        for line in read_file:
            if line:
                if begin_record_rewards and line[0] != 'Sum':
                    opt_train_rewards.append(float(line[-1]))
                if line[-1] == 'Reward':
                    begin_record_rewards = True
                end_line = line
        optimal_train_sum = float(end_line[-1])    

    env = ViewselEnv(input_file_dir, max_num_queries, max_num_views, support_frac, suffix)
    #env.seed(11)  #actions are deterministic based on seed 0
    agent = DQN(env=env, max_mem_chunk_len=max_mem_chunk_len, save_mem_type=save_mem_type)

    if semi_supervised:
        mem_fn = "greedy_memory"+suffix+".p"
        agent.memory = pickle.load(open(input_file_dir/mem_fn, 'rb'))
    print('# loaded samples: ', len(agent.memory))

    # if load_prev_model:
    #     if os.path.exists(input_file_dir/'norewrites_run_1_tr.model'):
    #         agent.model = load_model(input_file_dir/'norewrites_run_1_tr.model')
    #         agent.target_model = load_model(input_file_dir/'norewrites_run_1_tr.model')
    #         print('MODEL LOADED')

    # for test in test_samples:
    #     if test in samples:
    #         print('Test sample is in training set!')
    #         sys.exit()
    if num_episodes > 1:
        if write_train_file:
            if len(sys.argv) == 1:
                fn = vers + suffix+'train.txt'
                output_name = output_path/fn
            else:
                fn = vers + suffix+'train_run_' +str(sys.argv[1])+ '.txt'
                output_name = output_path/fn
            f = open(output_name, 'w')  #reset old file, so no need to manually delete it each run
            file = open(output_name, 'a')
    if write_test_file:
        fn=vers+'train_run_' +str(sys.argv[1]) + suffix+ '_test.txt'
        test_output_name = output_path/fn
        f = open(test_output_name, 'w')  #reset old file, so no need to manually delete it each run
        test_file = open(test_output_name, 'a')
        test_file.write('optimal sum: ' + str(optimal_sum) + ';  stop_thres: ' + str(stop_thres))
    fn = vers+"run_"+str(sys.argv[1])+ suffix+"_tr_test.csv"
    csv_name = output_path/fn
    fn = vers+"run_"+str(sys.argv[1])+ suffix+"_train_frac_opt.p"
    train_result_name = output_path/fn
    fn = vers+"run_"+str(sys.argv[1])+ suffix+"_test_frac_opt.p"
    test_result_name = output_path/fn
    fn = vers+"run_"+str(sys.argv[1])+ suffix+"_iters_per_ep.p"
    iters_fn = output_path/fn
    fn = vers+"run_"+str(sys.argv[1])+ suffix+"_all_train_rewards.p"
    all_train_fn = output_path/fn
    fn = vers+"run_"+str(sys.argv[1])+ suffix+"_all_test_fracopt.p"
    all_test_fn = output_path/fn
    fn = vers+"run_"+str(sys.argv[1])+ suffix+"_train_indivdiff_avgs.p"
    train_compare_name = output_path/fn
    fn = vers+"run_"+str(sys.argv[1])+ suffix+"_test_indivdiff_cols.p"
    test_compare_name = output_path/fn
    fn = vers+"run_"+str(sys.argv[1])+ suffix+"_rew_histos.p"
    histos_name = output_path/fn
    all_train_rewards = []
    all_test_fracopt = [] #track progress of any test workload
    all_rewards = []  #recorded in spreadsheet
    train_frac_opt = []
    test_frac_opt = []
    all_compare_cols = []
    train_compare_avgs = []
    all_histos = []
    continue_training = True
    if num_test_csv_samp_torec == 'all':
        num_test_csv_samp_torec = len(test_samples)

    ### Begin training ###
    start = time.time()
    num_iterations = 0
    iterations_per_ep = []
    for i_episode in range(1, num_episodes + 1): 
        ### TRAIN EPISODE ###
        agent.epsilon *= agent.epsilon_decay
        agent.epsilon = max(agent.epsilon_min, agent.epsilon)
        # ep_start = time.time()
        if i_episode % 1000 == 0 or i_episode == 1:
            print('Episode: ', i_episode)
            # print('eps: ', agent.epsilon)
        # if i_episode % 5000 == 0:
        #     end = time.time()
        #     hours, rem = divmod(end-start, 3600)
        #     minutes, seconds = divmod(rem, 60)
        #     seconds = round(seconds, 2)
        #     print("{:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))

        input_queries = random.choice(samples)
        old_state = env.new_ep(input_queries)
        if len(env.input_view_strs) == 0:  
            continue

        invalids = []  #store b/c diff # of views each ep
        # for ac in range(len(agent.env.input_view_strs), agent.env.max_num_views):
        #     invalids.append(ac)

        old_state = np.reshape(old_state, (1, env.state.shape[0], 1))
        if num_episodes > 1 and write_train_file:
            file.write('\nEpisode: ' + repr(i_episode)+ '\n')
            file.write('Epsilon: ' + repr(agent.epsilon)+ '\n')
            for q, qry in enumerate(input_queries):
                file.write('Input query #' + str(q) + ': ' + repr(qry) + '\n')
            #file.write('View order: ' + str(env.input_view_ids) + '\n')

        done = False
        for t in count():  #keeps on going forever until break
            action = agent.act(old_state, file = file)
            new_state, reward, done = env.step(action)
            new_state = np.reshape(new_state, (1, env.state.shape[0], 1))

            if memory_samp_num < max_memory_len:
                if save_mem_type == 'single':
                    fn_num = str(memory_samp_num) + '.p'
                    pickle.dump( [old_state, action, reward, new_state, done], open( memory_folder/fn_num, "wb" ) )  
                else:
                    if save_mem_type == 'chunks':
                        if len(agent.memory) >= max_mem_chunk_len:
                            begin_ind = int( mem_chunk_groupno * max_mem_chunk_len)
                            end_ind = int( (mem_chunk_groupno + 1) * max_mem_chunk_len) - 1
                            fn_num = str(begin_ind) +'_to_' + str(end_ind) + '.p'
                            pickle.dump( agent.memory, open( memory_folder/fn_num, "wb" ) )  
                            mem_chunk_groupno += 1
                            agent.memory = deque(maxlen=max_mem_chunk_len + 1)
                    agent.remember(old_state, action, reward, new_state, done)
                memory_samp_num += 1
            
            agent.replay(memory_folder, memory_samp_num, mem_chunk_groupno)
            agent.target_train() 
            num_iterations += 1
            old_state = new_state

            if done and num_episodes > 1 and write_train_file:
                record_ep(file, env)
                break
            elif done:
                break

        if i_episode % model_save_interval == 0:
            #fn = model_name+ suffix +'_tr_' + str(i_episode) + '_eps.model'
            fn = model_name+ suffix +'_tr.model'
            agent.save_model(output_path/fn)
            # pickle.dump( agent.memory, open( output_path/"memory.p", "wb" ) )
        ### end episode ###

        ### Eval Training set ###
        if eval_trainset and (i_episode % train_interval == 0 or i_episode == 1):
            model_rewards = []
            # if write_test_file:
            #     test_file.write('\n'+ '<'*130 +'\n')
            #     test_file.write(str(i_episode) + ' Training Episodes\n')

            #     end = time.time()
            #     hours, rem = divmod(end-start, 3600)
            #     minutes, seconds = divmod(rem, 60)
            #     seconds = round(seconds, 2)
            #     test_file.write(str(hours) + ':' + str(minutes) + ':' + str(seconds))
            
            ### Begin testing each TRAINING sample ###
            for num, input_queries in enumerate(samples):
                if num % (len(samples)/5) == 0:
                    print(num, len(samples))
                # if write_test_file:
                #     test_file.write('\nSample ' + str(num) + '\n')
                #     for q, qry in enumerate(input_queries):
                #         test_file.write('Input query #' + str(q) + ': ' + repr(qry) + '\n')

                old_state = env.new_ep(input_queries)
                if len(env.input_view_strs) == 0:  
                    model_rewards.append(0)
                    continue
                old_state = np.reshape(old_state, (1, env.state.shape[0], 1))
                done = False
                for t in count():  #keeps on going forever until break
                    action, preds, inds_of_highest = agent.test_act(old_state)
                    new_state, reward, done = env.step(action)
                    new_state = np.reshape(new_state, (1, env.state.shape[0], 1))
                    old_state = new_state

                    # if write_test_file and not done and env.strikes == 0:
                    #     test_file.write('\nAction#: ' + repr(action) + '\n')
                    #     test_file.write('\nmaint cost: ' + repr(env.maint_cost) + '\n')
                    #     test_file.write(repr(env.maint_cost_lst) + '\n')
                    #     test_file.write('eval cost: ' + repr(env.eval_cost) + '\n')
                    #     test_file.write('total cost: ' + repr(env.new_cost) + '\n')   

                    if done:
                        # if write_test_file:
                        #     record_ep(test_file, env)
                        model_rewards.append(round(env.reward, 4))
                        break

            compare_col = []
            for RL_reward, opt_reward in zip(model_rewards, opt_train_rewards):
                if opt_reward != 0:
                    compare_col.append(RL_reward/opt_reward)
                else:
                    # compare_col.append(RL_reward / (RL_reward - 0.1))
                    compare_col.append(1)
            compare_avg = sum(compare_col) / len(compare_col)
            train_compare_avgs.append(compare_avg)

            # all_train_rewards.append(model_rewards)  
            frac_opt = ( sum(model_rewards) / optimal_train_sum )
            print('Train Frac of opt: ', frac_opt)
            train_frac_opt.append(frac_opt)
            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            seconds = round(seconds, 2)
            print("{:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))

            # pickle.dump( all_train_rewards, open( all_train_fn, "wb" ) )
            pickle.dump( train_compare_avgs, open( train_compare_name, "wb" ) )

        ### Eval Testing set ###
        qry_to_numviews_sel = {}
        if i_episode % test_interval == 0 or i_episode == 1:
            iterations_per_ep.append(num_iterations)
            model_rewards = []
            samp_result_histo = [0] * 12
            if write_test_file:
                test_file.write('\n'+ '<'*130 +'\n')
                test_file.write(str(i_episode) + ' Training Episodes\n')
            
            ### Begin testing each TEST sample ###
            for num, input_queries in enumerate(test_samples):
                # if num % (len(test_samples)/5) == 0:
                #     print(num, len(test_samples))
                old_state = env.new_ep(input_queries)
                if len(env.input_view_strs) == 0:  
                    model_rewards.append(0)
                    continue

                # if write_test_file and (num == 1):
                if write_test_file:
                    test_file.write('\nSample ' + str(num) + '\n')
                    for q, qry in enumerate(input_queries):
                        test_file.write('Input query #' + str(q) + ': ' + repr(qry) + '\n')
                    for q, qry in enumerate(env.input_view_strs):
                        test_file.write('View #' + str(q) + ': ' + repr(qry) + '\n')
                old_state = np.reshape(old_state, (1, env.state.shape[0], 1))
                done = False
                for t in count():  #keeps on going forever until break
                    action, preds, inds_of_highest = agent.test_act(old_state)
                    new_state, reward, done = env.step(action)
                    new_state = np.reshape(new_state, (1, env.state.shape[0], 1))
                    old_state = new_state

                    # if write_test_file and not done and env.strikes == 0:
                    # if write_test_file and (num == 1):
                    if write_test_file:
                        test_file.write(repr(preds) + '\n')
                        test_file.write(repr(inds_of_highest) + '\n')
                        test_file.write('pos ' + repr(action) + ': ')
                        test_file.write(repr(agent.env.input_view_strs[action]) + '\n')
                        # test_file.write('\nAction#: ' + repr(action) + '\n')
                        test_file.write('\nmaint cost: ' + repr(env.maint_cost) + '\n')
                        test_file.write(repr(env.maint_cost_lst) + '\n')
                        test_file.write('eval cost: ' + repr(env.eval_cost) + '\n')
                        test_file.write('total cost: ' + repr(env.new_cost) + '\n')   
                        # record_ep(test_file, env)

                    if done:
                        if write_test_file:
                            record_ep(test_file, env)
                    if done:
                        model_rewards.append(round(env.reward, 4))
                        qry_to_numviews_sel[num] = len(env.maint_cost_lst)
                        break
            frac_opt = ( sum(model_rewards) / optimal_sum )
            compare_col = []
            for RL_reward, opt_reward in zip(model_rewards, opt_test_rewards):
                if opt_reward != 0:
                    compare_col.append(RL_reward/opt_reward)
                else:
                    # compare_col.append(RL_reward / (RL_reward - 0.1))
                    compare_col.append(1)

            model_sum = sum(model_rewards)
            compare_avg = sum(compare_col) / len(compare_col)
            avgnum_views_sel = sum(qry_to_numviews_sel.values())/len(qry_to_numviews_sel)
            samps_better = ''
            for samp_num, val in enumerate(compare_col):
                if val < 1:
                    samp_result_histo[math.floor(val*10)] += 1
                elif val == 1:
                    samp_result_histo[10] += 1
                else:
                    samp_result_histo[-1] += 1
                    if len(samps_better.split(',')) < 5:
                        samps_better += str(samp_num) + ','

            all_test_fracopt.append(compare_col)  
            pickle.dump( all_test_fracopt, open( all_test_fn, "wb" ) )
            model_rewards=model_rewards[:num_test_csv_samp_torec]
            compare_col=compare_col[:num_test_csv_samp_torec]

            compare_col.append(compare_avg)
            model_rewards.append(model_sum)  #to record sum in all_rewards
            model_rewards.append(frac_opt)  #to record frac_opt in all_rewards
            model_rewards.append(avgnum_views_sel)
            all_compare_cols.append(compare_col)
            all_histos.append(samp_result_histo)
            if eval_trainset and (i_episode % train_interval == 0 or i_episode == 1):
                model_rewards.append(train_frac_opt[-1])
            else:
                model_rewards.append('--')

            #print('Episode: ', i_episode)
            print('Test Frac of opt: ', frac_opt)
            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            seconds = round(seconds, 2)
            print("{:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))
            model_rewards.append(str(int(hours)) + ' hrs, ' + str(int(minutes)) + ' mins, ' + str(seconds) +' secs')

            model_rewards.append(samps_better)

            all_rewards.append(model_rewards)  #to record in spreadsheet
            test_frac_opt.append(frac_opt)  #to plot

            pickle.dump( train_frac_opt, open( train_result_name, "wb" ) )
            pickle.dump( test_frac_opt, open( test_result_name, "wb" ) )
            pickle.dump( iterations_per_ep, open( iters_fn, "wb" ) )
            transpose_all_rewards = np.array(all_rewards).T.tolist()
            transpose_compare_cols = np.array(all_compare_cols).T.tolist()
            transpose_histos = np.array(all_histos).T.tolist()
            pickle.dump( transpose_compare_cols, open( test_compare_name, "wb" ) )
            pickle.dump( all_histos, open( histos_name, "wb" ) )

            rownames = ['samp ' + str(i) for i in range(len(transpose_all_rewards)-6)] + ['sum', 'frac of '+opt_algo_type_short, 'avg#views', 'train frac of '+opt_algo_type_short, 'time', 'samps >100%']
            for i in range(len(transpose_all_rewards)):
                if i < len(transpose_all_rewards)-6:
                    transpose_all_rewards[i] = [opt_test_rewards[i]] + [rownames[i]] + transpose_all_rewards[i]
                elif i == len(transpose_all_rewards)-6:
                    transpose_all_rewards[i] = [optimal_sum] + [rownames[i]] + transpose_all_rewards[i]
                else:  #last few rows
                    transpose_all_rewards[i] = ['']+[rownames[i]] + transpose_all_rewards[i]
            rownames_2 = ['samp ' + str(i) for i in range(len(transpose_compare_cols)-1)] + ['Avg']
            for i in range(len(transpose_compare_cols)):
                transpose_compare_cols[i] = ['']+[rownames_2[i]] + transpose_compare_cols[i]
            rownames_3 = [str(i*10)+'% to '+str(i*10 +9)+'%' for i in range(10)] + ['100%', '> 100%']
            for i in range(len(transpose_histos)):
                transpose_histos[i] = ['']+[rownames_3[i]] + transpose_histos[i]

            with open(csv_name, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([opt_algo_type_short+' sum: ', optimal_sum, '', 'stop thres: ', stop_thres])
                writer.writerow(['Support frac: ', support_frac, '', 'max num views: ', max_num_views, '', 'max_memory_len: ', max_memory_len, '', 'save_mem_type', save_mem_type])                
                writer.writerow(['Epsdecay: ', agent.epsilon_decay, '', 'lr: ', agent.learning_rate, '', 'lr decay: ', agent.decay_rate, '', 'Batchsize: ', agent.batch_size, '', 'Gamma: ', agent.gamma, '', 'Tau: ', agent.tau])                
                writer.writerow(['Rec Samps: ', num_test_csv_samp_torec, ' ', 'Total Test Samps: ', len(test_samples), '',
                    'Total Train Samps: ', len(samples)])
                writer.writerow([])
                writer.writerow(['','Iters ']+[str(i) for i in iterations_per_ep] )
                writer.writerow([opt_algo_type[:9]]+['Eps ']+['1']+[str(i * test_interval) for i in range(1,len(all_rewards))])
                writer.writerows(transpose_all_rewards)
                writer.writerow([])
                writer.writerow(['Indiv Samp Frac over '+opt_algo_type_short])
                writer.writerows(transpose_compare_cols)
                writer.writerow([])
                writer.writerow(['Histogram: ', 'Bin', 'Freq'])
                writer.writerows(transpose_histos)

            if frac_opt >= stop_thres:
                stop_strike += 1
            else:
                stop_strike = 0
            if stop_strike == stop_strike_limit:
                continue_training = False 
            ### end testing each sample ###
            
            # num_item_sizes = 5
            # mem = agent.memory
            # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
            #                          key= lambda x: -x[1])[:num_item_sizes]:
            #     if name == 'mem':
            #         print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        if continue_training == False:
            break  #stop training
        ### end test validation set ###

    ### end training ###

    ### Save model and analysis items ###
    if num_episodes < 1:
        fn=model_name+ suffix +'_untr.model'
        agent.save_model(output_path/fn)
    else:
        fn = model_name+ suffix+'_tr.model'
        agent.save_model(output_path/fn)
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        seconds = round(seconds, 2)
        if write_train_file:
            file.write(str(hours) + ':' + str(minutes) + ':' + str(seconds))

        # if test_frac_opt:
        #     fig = plt.figure()
        #     plt.plot(range(len(test_frac_opt)), test_frac_opt, marker='o')
        #     plt.xlabel('Eps')
        #     plt.ylabel('Test frac_opt')
        #     fn='run_' + str(sys.argv[1]) + '_test_frac_opt.png'
        #     fig.savefig(output_path/fn)
        #     plt.show()
        #     plt.close()

        # #file2 = open('query_to_nums.txt', 'w')
        # for q, qry in enumerate(agent.qry_rewards_dict.keys()):
        #     #file2.write(str(q) + ': ' + qry + '\n')
        #     fig = plt.figure()
        #     rewards = agent.qry_rewards_dict[qry]
        #     plt.plot(range(len(rewards)), rewards, marker='o')
        #     plt.xlabel('Episode')
        #     plt.ylabel('Reward')
        #     fig.savefig(output_path/'query'+str(q)+'.png')
        #     plt.close()

        # fig = plt.figure()
        # plt.plot(range(len(agent.action_rank_plot)), agent.action_rank_plot, marker='o')
        # plt.xlabel('Duration')
        # plt.ylabel('Qvalue Rank')
        # fig.savefig(output_path/'action_10_qvals_rank.png')
        # plt.show()
        # plt.close()

        # fig = plt.figure()
        # plt.plot(range(len(agent.action_plot)), agent.action_plot, marker='o')
        # plt.xlabel('Duration')
        # plt.ylabel('Qvalue')
        # fig.savefig(output_path/'action_10_qvals.png')
        # plt.show()
        # plt.close()

if __name__ == "__main__":
  try:
    main()
  except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
    # if del_mem_folder:
    #     shutil.rmtree(memory_folder)