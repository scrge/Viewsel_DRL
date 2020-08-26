#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import math, random, pickle, copy, re, csv, time, os
from functools import reduce
import itertools
import pdb, traceback, sys
import numpy as np
import networkx as nx
from pathlib import Path

class ViewselEnv():
    def __init__(self, input_file_dir, max_num_views, support_frac, suffix):
        self.all_queries = pickle.load(open(input_file_dir/'all_queries.p', 'rb'))  #all possible queries
        self.all_query_freqs = pickle.load(open(input_file_dir/'query_freqs.p', 'rb'))
        self.reln_attr_index = pickle.load(open(input_file_dir/'reln_attr_index.p', 'rb'))
        self.RA = len(self.reln_attr_index)  #should be specific DB to DB
        self.reln_sizes = pickle.load(open(input_file_dir/'reln_sizes.p', 'rb'))
        self.join_selectivities = pickle.load(open(input_file_dir/'join_selectivities.p', 'rb'))
        self.max_num_views = max_num_views
        self.support_frac = support_frac

        if os.path.exists(input_file_dir/'base_reln_update_freqs.p'):
            self.base_reln_update_freqs = pickle.load(open(input_file_dir/'base_reln_update_freqs.p', 'rb'))  
        else:
            self.base_reln_update_freqs = [1 for i in range(len(self.reln_sizes))]

    def _generate_views(self):
        #start_time = time.time()
        lst_views = []
        view_to_freq = {}
        for query in self.input_query_strs:
            query = list(query)
            views_by_num_edges = {1:[]} # num_edges_in_view : views
            for edge in query:
                if frozenset(edge) in view_to_freq:
                    freq_val = view_to_freq[frozenset(edge)]
                else:
                    freq_val = 0
                    for qry in self.input_query_strs:
                        if set([edge]).issubset(qry):
                            ind = self.all_queries.index(qry)
                            freq_val += self.all_query_freqs[ind]
                    view_to_freq[frozenset(edge)] = freq_val   
                if freq_val > self.support:
                    views_by_num_edges[1].append(set([edge]))
            edge_to_neighbors = {}  # edge : all edges that neighbor the key edge's RELATIONS
            for edge_1 in query:
                edge_1 = tuple(edge_1)
                edge_1_relations = [edge_1[0].split('.')[0], edge_1[1].split('.')[0]]
                for reln in edge_1_relations:
                    for edge_2 in query:
                        edge_2 = tuple(edge_2)
                        edge_2_relations = [edge_2[0].split('.')[0], edge_2[1].split('.')[0]]
                        if reln in edge_2_relations and edge_1 != edge_2:
                            if frozenset(edge_1) in edge_to_neighbors:
                                edge_to_neighbors[frozenset(edge_1)].append(frozenset(edge_2))  
                            else:
                                edge_to_neighbors[frozenset(edge_1)] = [frozenset(edge_2)]

            # add only edges to new view
            if len(query) > 5:
                end_len = 5
            else:
                end_len = len(query)
            for num_edges in range(2, end_len):
                #print(num_edges, len(query))
                views_by_num_edges[num_edges] = []
                prev_lvl = views_by_num_edges[num_edges - 1]
                for ii, lower_view in enumerate(prev_lvl):
                    #print(num_edges, ii, len(prev_lvl), len(views_by_num_edges[num_edges]))
                    lower_view_neighbors = []
                    for edge in list(lower_view):
                        lower_view_neighbors += edge_to_neighbors[edge]
                    lower_view_neighbors = list(set(lower_view_neighbors))
                    for neighbor in lower_view_neighbors:
                        new_view = lower_view.union(set([neighbor]))
                        if frozenset(new_view) in view_to_freq:
                            freq_val = view_to_freq[frozenset(new_view)]
                        else:
                            freq_val = 0
                            for qry in self.input_query_strs:
                                if set(new_view).issubset(qry):
                                    ind = self.all_queries.index(qry)
                                    freq_val += self.all_query_freqs[ind]
                            view_to_freq[frozenset(new_view)] = freq_val    #may not be good for this qry but good for next
                        #check if new view freq > support threshold
                        if freq_val > self.support and new_view not in views_by_num_edges[num_edges]:
                            views_by_num_edges[num_edges].append(new_view)
                            #add to top X views.
                #keep only top views
                new_views_by_num_edges = []
                for view in views_by_num_edges[num_edges]:
                    if view not in new_views_by_num_edges:
                        new_views_by_num_edges.append(view)

                lst_freq_vals = []
                for view in new_views_by_num_edges:
                    freq_val = 0
                    for qry in self.input_query_strs:
                        ind = self.all_queries.index(qry)
                        if set(view).issubset(qry):
                            freq_val += self.all_query_freqs[ind]
                    lst_freq_vals.append(freq_val)

                view_sizes = []
                view_joinsel_prods = []
                for view in new_views_by_num_edges:
                    view_size = 1
                    view_joinsel_prod = 1
                    relns_so_far = []
                    for edge in view:
                        edge = tuple(edge)
                        left_reln = edge[0].split('.')[0]
                        right_reln = edge[1].split('.')[0]
                        join_sel = self.join_selectivities[edge]
                        view_size *= join_sel
                        view_joinsel_prod *= join_sel
                        if left_reln not in relns_so_far:
                            view_size *= self.reln_sizes[left_reln]
                            relns_so_far.append(left_reln)
                        if right_reln not in relns_so_far:
                            view_size *= self.reln_sizes[right_reln]
                            relns_so_far.append(right_reln)
                    view_sizes.append(view_size)
                    view_joinsel_prods.append(view_joinsel_prod)

                lst_tuples = list(zip(new_views_by_num_edges, lst_freq_vals, view_sizes, view_joinsel_prods))
                sorted_lst_tuples = sorted(lst_tuples, key=lambda tup: (tup[1], -tup[2], -tup[3])) 
                sorted_lst_tuples.reverse()
                sorted_lst_views = []
                for tup in sorted_lst_tuples:
                    sorted_lst_views.append(tup[0])
                # views_by_num_edges[num_edges] = sorted_lst_views[:self.max_num_views*10]  
                views_by_num_edges[num_edges] = sorted_lst_views[:self.max_num_views] 
                #
            for v_lst in views_by_num_edges.values():
                lst_views.extend(v_lst)

        new_lst_views = []
        for view in lst_views:
            if view not in new_lst_views:
                new_lst_views.append(view)
        lst_views = new_lst_views

        lst_freq_vals = []
        for view in lst_views:
            freq_val = 0
            for qry in self.input_query_strs:
                ind = self.all_queries.index(qry)
                if set(view).issubset(qry):
                    freq_val += self.all_query_freqs[ind]
            lst_freq_vals.append(freq_val)

        view_sizes = []
        view_joinsel_prods = []
        for i, view in enumerate(lst_views):
            view_size = 1
            view_joinsel_prod = 1
            relns_so_far = []
            for edge in view:
                edge = tuple(edge)
                left_reln = edge[0].split('.')[0]
                right_reln = edge[1].split('.')[0]
                join_sel = self.join_selectivities[edge]
                view_size *= join_sel
                view_joinsel_prod *= join_sel
                if left_reln not in relns_so_far:
                    view_size *= self.reln_sizes[left_reln]
                    relns_so_far.append(left_reln)
                if right_reln not in relns_so_far:
                    view_size *= self.reln_sizes[right_reln]
                    relns_so_far.append(right_reln)
            view_sizes.append(view_size)
            view_joinsel_prods.append(view_joinsel_prod)

        lst_tuples = list(zip(lst_views, lst_freq_vals, view_sizes, view_joinsel_prods))
        #prio: most freq, lowest cost, lowest size, product of joinsels
        sorted_lst_tuples = sorted(lst_tuples, key=lambda tup: (tup[1], -tup[2], -tup[3])) 
        sorted_lst_tuples.reverse()
        sorted_lst_views = []
        for tup in sorted_lst_tuples:
            sorted_lst_views.append(tup[0])
        lst_views = sorted_lst_views[:self.max_num_views]  
        #print("--- %s seconds ---" % (time.time() - start_time))
        return lst_views

    def new_ep(self, input_query_strs, qry_to_views):    #Reset the state of the environment with new inputs
        self.input_queries = np.zeros((self.max_num_queries, self.RA, self.RA))
        self.input_views = np.zeros((self.max_num_views, self.RA, self.RA))
        self.views_selected = np.zeros(self.max_num_views)
        self.query_rewrites = np.zeros((self.max_num_queries, self.max_num_views))

        self.input_query_strs = input_query_strs
        self.num_input_qrys = len(self.input_query_strs)

        if tuple(input_query_strs) in qry_to_views:
            view_tuple = qry_to_views[tuple(input_query_strs)]
            self.input_view_strs = view_tuple[0]
            self.view_sizes = view_tuple[1]
            self.view_costs = view_tuple[2]
            self.relns_of_view = view_tuple[3]
            self.relns_of_input_queries = view_tuple[4]
            self.qry_freqs = view_tuple[5]
        else:
            self.relns_of_input_queries = [0]*len(self.input_query_strs)
            self.qry_freqs = []
            for i, iq in enumerate(self.input_query_strs):
                relns_of_query = []
                for edge in iq:
                    edge = tuple(edge)
                    left_reln = edge[0].split('.')[0]
                    right_reln = edge[1].split('.')[0]
                    if left_reln not in relns_of_query:
                        relns_of_query.append(left_reln)
                    if right_reln not in relns_of_query:
                        relns_of_query.append(right_reln)
                self.relns_of_input_queries[i] = relns_of_query
                ind = self.all_queries.index(iq)
                self.qry_freqs.append(self.all_query_freqs[ind])
            self.support = sum(sorted(self.qry_freqs)[:math.floor(len(self.qry_freqs) * self.support_frac)])  
            self.input_view_strs = self._generate_views()       

            self.view_sizes = {} #view position ID : size, for selected views
            self.view_costs = {}
            self.relns_of_view = {}
            for i, view in enumerate(self.input_view_strs):
                view_cost = 0
                view_size = 1
                relns_of_view = []
                for edge in view:
                    edge = tuple(edge)
                    left_reln = edge[0].split('.')[0]
                    right_reln = edge[1].split('.')[0]
                    join_sel = self.join_selectivities[edge]
                    view_size *= join_sel
                    if left_reln not in relns_of_view:
                        view_size *= self.reln_sizes[left_reln]
                        view_cost += self.reln_sizes[left_reln]
                        relns_of_view.append(left_reln)
                    if right_reln not in relns_of_view:
                        view_size *= self.reln_sizes[right_reln]
                        view_cost += self.reln_sizes[right_reln]
                        relns_of_view.append(right_reln)
                u_freq = 0
                for reln in relns_of_view:
                    u_freq += self.base_reln_update_freqs[reln]
                u_freq = u_freq / len(relns_of_view)
                self.view_sizes[i] = view_size
                self.view_costs[i] = u_freq * view_cost
                self.relns_of_view[i] = sorted(relns_of_view)

        for i, qry in enumerate(self.input_query_strs):  #represent self.input_queries in vector
            for edge in qry:
                edge = tuple(edge)
                row_ind = self.reln_attr_index.index(edge[0])
                col_ind = self.reln_attr_index.index(edge[1])
                self.input_queries[i, row_ind, col_ind] = 1
                self.input_queries[i, col_ind, row_ind] = 1  #symmetric matrix
        for i, view in enumerate(self.input_view_strs):  #represent self.input_queries in vector
            view = list(view)
            for edge in view:
                edge = tuple(edge)
                row_ind = self.reln_attr_index.index(edge[0])
                col_ind = self.reln_attr_index.index(edge[1])
                self.input_views[i, row_ind, col_ind] = 1
                self.input_views[i, col_ind, row_ind] = 1  #symmetric matrix
        self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), self.views_selected, self.query_rewrites.flatten()), axis=0)
        self.state = self.state.astype(int)

        # Store what the agent tried
        self.lst_relns_used_in_rewriting = [[]]*self.num_input_qrys
        self.views_tried = [] #list of actions mapped to views
        self.done = False
        self.rewrites_dict = {i:[] for i in range(self.num_input_qrys)}  #rewrite is a list of view_ids
        self.maint_cost_lst = {}
        self.init_cost, self.qry_proc_costs, x = self._get_eval_cost(self.rewrites_dict, None, 'all')
        self.new_cost = self.init_cost
        self.reward = 0
        return self.state

    def step(self, file):
        action_to_rewards = {}
        action_to_results = {}
        for view_id in range(len(self.input_view_strs)):
            if view_id in self.views_tried:
                continue
            temp_views_tried = copy.deepcopy(self.views_tried)  #temp copy bc will discard if not better
            temp_views_tried.append(view_id)  #need a temp bc this has the potential view candidate
            #optimally rewrite queries using selected views
            temp_rewrites_dict = copy.deepcopy(self.rewrites_dict)
            for q_ind, qry in enumerate(self.input_query_strs):
                if self.input_view_strs[view_id].issubset(qry):
                    no_sub_or_super = True
                    subviews_of_action = []
                    for sel_view in temp_rewrites_dict[q_ind]:
                        is_sub = self.input_view_strs[view_id].issubset(self.input_view_strs[sel_view]) #v3
                        is_super = self.input_view_strs[sel_view].issubset(self.input_view_strs[view_id]) #v3
                        # isdisjoint = self.input_view_strs[sel_view].isdisjoint(self.input_view_strs[view_id])
                        # if is_sub or is_super:
                        if is_sub:
                        # if is_super:
                        # if not isdisjoint:
                        # if not isdisjoint and is_sub and is_super:
                            no_sub_or_super = False  #don't add view if it's subview of existing view in rewriting
                        if is_super: 
                            subviews_of_action.append(sel_view)
                    #if new view is superview of existing views in rewriting, must remove them to add
                    #ONLY REPLACE IF IT IMPROVES COST
                    # if no_sub_or_super:
                    #     temp_rewrites_dict[q_ind].append(view_id)
                    if subviews_of_action and no_sub_or_super:                     
                        temp_temp_rewrites_dict = copy.deepcopy(temp_rewrites_dict)
                        for view in subviews_of_action:
                            temp_temp_rewrites_dict[q_ind].remove(view)
                        temp_temp_rewrites_dict[q_ind].append(view_id)
                        new_eval_cost = self._get_eval_cost(temp_temp_rewrites_dict, view_id, q_ind)
                        if self.qry_proc_costs[q_ind] > new_eval_cost:
                            for view in subviews_of_action:
                                temp_rewrites_dict[q_ind].remove(view)
                            temp_rewrites_dict[q_ind].append(view_id)      
                    elif no_sub_or_super:                        
                        temp_temp_rewrites_dict = copy.deepcopy(temp_rewrites_dict)
                        temp_temp_rewrites_dict[q_ind].append(view_id)
                        new_eval_cost = self._get_eval_cost(temp_temp_rewrites_dict, view_id, q_ind)
                        if self.qry_proc_costs[q_ind] > new_eval_cost:
                            temp_rewrites_dict[q_ind].append(view_id)

            #remove any views not used in rewriting
            #for each view, check if it's in at least on rewriting. if not, remove it.
            for view in temp_views_tried:
                remove_bool = True
                for rewrites_lst in temp_rewrites_dict.values():
                    if view in rewrites_lst:
                        remove_bool = False
                if remove_bool:
                    temp_views_tried.remove(view)

            # if set(temp_views_tried) == set([0,3]):
            #      pdb.set_trace()

            #check if new rewrites of potential new state beget lower cost than before
            #(new_reward, new_cost, lst_relns_used_in_rewriting, maint_cost_lst, qry_proc_costs) 
            reward_outputs = list(self._get_reward(temp_rewrites_dict, temp_views_tried, view_id))
            reward_outputs.append(temp_views_tried)
            reward_outputs.append(temp_rewrites_dict)
            reward_outputs.append(self.maint_cost)
            reward_outputs.append(self.eval_cost)
            reward_outputs.append(self.new_cost)
            action_to_results[view_id] = copy.deepcopy(reward_outputs)
            action_to_rewards[view_id] = action_to_results[view_id][0]

        file.write(repr(action_to_rewards) + '\n')

        best_view = max(action_to_rewards, key=action_to_rewards.get)

        self.views_selected[best_view] = 1  #write selview into state tensor    
        for q_ind in action_to_results[best_view][6]:   
            for v_ind in action_to_results[best_view][6][q_ind]:    
                self.query_rewrites[q_ind][v_ind] = 1   
        self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), self.views_selected, self.query_rewrites.flatten()), axis=0)  
        self.state = self.state.astype(int)

        #after finding best view, only select view if it's used in at least ONE rewriting.
        #if it's not used in at least one view, rewritings won't change, and the reward will never be greater.
        if action_to_rewards[best_view] > self.reward:
            self.views_tried = copy.deepcopy(action_to_results[best_view][5])
            self.rewrites_dict = copy.deepcopy(action_to_results[best_view][6])
            self.reward = copy.deepcopy(action_to_results[best_view][0])
            self.new_cost = copy.deepcopy(action_to_results[best_view][1])
            self.lst_relns_used_in_rewriting = copy.deepcopy(action_to_results[best_view][2])
            self.maint_cost_lst = copy.deepcopy(action_to_results[best_view][3])
            self.qry_proc_costs = copy.deepcopy(action_to_results[best_view][4])            
            file.write('\nmaint cost: ' + repr(action_to_results[best_view][7]) + '\n')
            file.write('eval cost: ' + repr(action_to_results[best_view][8]) + '\n')
            file.write('total cost: ' + repr(action_to_results[best_view][9]) + '\n')
        else:   
            self.done = True    
            self._record_choice(file)   
            return self.state, best_view, action_to_rewards[best_view], self.done   
        if len(self.views_tried) == len(self.input_view_strs):  
            self.done = True    
            self._record_choice(file)   
        return self.state, best_view, action_to_rewards[best_view], self.done

    def _get_eval_cost(self, rewrites_dict, new_view, qry_num):
        processing_cost = 0 #cost of repr all queries
        qry_proc_costs = {}
        temp_relns_in_rewriting = copy.deepcopy(self.lst_relns_used_in_rewriting)
        if qry_num == 'all':
            qry_lst = range(self.num_input_qrys)
        else:
            qry_lst = [qry_num]
        if new_view != None:
            for q_ind in qry_lst:
                if new_view in rewrites_dict[q_ind]:
                    temp_relns_in_rewriting[q_ind] = list(set(temp_relns_in_rewriting[q_ind] + \
                        self.relns_of_view[new_view]))

        for q_ind in qry_lst:
            rewrite_lst_views_relns = []
            rewrite_lst_view_sizes = [] #sizes of views in this query's rewrite
            for v in rewrites_dict[q_ind]:
                rewrite_lst_view_sizes.append(self.view_sizes[v])
                rewrite_lst_views_relns.append(self.relns_of_view[v])
            #get which relns are missing from views used in rewriting so far   
            uncovered_relns = list(set(self.relns_of_input_queries[q_ind]) - set(temp_relns_in_rewriting[q_ind]))
            for reln in uncovered_relns:
                rewrite_lst_view_sizes.append(self.reln_sizes[reln])
                rewrite_lst_views_relns.append([reln])

            iq_copy = list(copy.deepcopy(self.input_query_strs[q_ind]))
            iq_copy = [tuple(edge) for edge in iq_copy]
            if len(rewrite_lst_views_relns) > 1:
                total_intermediate_sizes = 0
                joinsels_interm_qry = []
                lst_tuples = list(zip(rewrite_lst_views_relns, rewrite_lst_view_sizes, \
                    [rel[0] for rel in rewrite_lst_views_relns]))
                views_to_join = sorted( lst_tuples, key=lambda tup: (tup[1], tup[2]) ) 
                views_to_join = [x[0] for x in views_to_join]
                covered_relns_so_far = views_to_join[0]
                del views_to_join[0]
                while views_to_join:
                    k = 0
                    join_view_bool = False
                    while k < len(views_to_join):
                        temp_covered_relns = list(set(covered_relns_so_far + views_to_join[k] ))
                        iq_copy_copy = copy.deepcopy(iq_copy)
                        for edge in iq_copy_copy:
                            left_reln = edge[0].split('.')[0]
                            right_reln = edge[1].split('.')[0]
                            if (left_reln in temp_covered_relns) and (right_reln in temp_covered_relns):
                                joinsels_interm_qry.append(self.join_selectivities[edge] ) 
                                iq_copy.remove(edge) 
                                covered_relns_so_far = copy.deepcopy(temp_covered_relns)
                                join_view_bool = True
                        if join_view_bool:
                            break
                        elif iq_copy == []:
                            k = 0
                            break
                        else:  #iq_copy unchanged
                            k += 1
                    covered_reln_sizes = [self.reln_sizes[reln] for reln in covered_relns_so_far]
                    total_intermediate_sizes += reduce(lambda x, y: x*y, covered_reln_sizes + joinsels_interm_qry) 
                    del views_to_join[k]
            else: #view = query
                covered_reln_sizes = [self.reln_sizes[reln] for reln in rewrite_lst_views_relns[0]]
                joinsels_interm_qry = [self.join_selectivities[edge] for edge in iq_copy]
                total_intermediate_sizes = reduce(lambda x, y: x*y, covered_reln_sizes + joinsels_interm_qry)

            qry_proc_costs[q_ind] = self.qry_freqs[q_ind] * \
                (sum(rewrite_lst_view_sizes) + total_intermediate_sizes )
            processing_cost += qry_proc_costs[q_ind]
        if qry_num == 'all':
            return processing_cost, qry_proc_costs, temp_relns_in_rewriting
        else:
            return processing_cost

    def _get_reward(self, rewrites_dict, views_tried, new_view):
        maint_cost_lst = {}
        maint_cost = 0
        for v in views_tried: #get views selected
            maint_cost += self.view_costs[v]
            maint_cost_lst[v] = self.view_costs[v]
        eval_cost, qry_proc_costs, temp_relns_in_rewriting = \
            self._get_eval_cost(rewrites_dict, new_view, 'all')
        new_cost = (maint_cost + eval_cost)
        if self.init_cost - new_cost > 0:
            reward = (self.init_cost - new_cost)/(self.init_cost)
        else:
            reward = -1
        self.maint_cost = maint_cost
        self.eval_cost = eval_cost
        self.new_cost = new_cost
        return reward, new_cost, temp_relns_in_rewriting, maint_cost_lst, qry_proc_costs

    def _record_choice(self, file):
        for v in self.maint_cost_lst.keys():
            file.write(repr(self.input_view_strs[v]) + ', Maint cost: ' + repr(self.maint_cost_lst[v]) +'\n')
        file.write('Rewrites:\n')
        for x in range(self.num_input_qrys):
            file.write('Query ' + repr(x)+ ', ')
            file.write('Proc cost: ' + repr(self.qry_proc_costs[x])+ '\n')
            for v in self.rewrites_dict[x]:  #each view is a list of edges
                file.write(repr(self.input_view_strs[v])+ '\n')
        file.write('Default Cost: ' + str(self.init_cost) + '\n')
        file.write('reward: ' + repr(self.reward) + '\n')

    def seed(self, seed):
        random.seed(seed)
        np.random.seed

    def get_views(self):
        return (self.input_view_strs, self.view_sizes, self.view_costs, self.relns_of_view,
                self.relns_of_input_queries, self.qry_freqs)

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def main():
    #sys.argv: [1]- k, [2]-split # : from 1 to k, [3]- 0 is train, 1 is test
    # dataset_name = 'Dataset60'
    dataset_name = 'JOB_6_2'
    max_num_views = 10
    support_frac = 0.25
    max_num_queries = 5
    memory  = deque(maxlen=2000000)

    input_file_dir = Path.cwd().parent / "datasets" / dataset_name 
    if len(sys.argv) < 3:
        if int(sys.argv[1]) == 0:
            train_or_test = 'train'
        else:
            train_or_test = 'test'
        suffix = ''
    else:
        if int(sys.argv[3]) == 0:
            train_or_test = 'train'
        else:
            train_or_test = 'test'

    if len(sys.argv) < 3:
        begin_ind = 1
        end_ind = 1
        if train_or_test == 'train':
            output_name = 'viewsel_train_greedy_newcostmodel'
            samples = pickle.load(open(input_file_dir/'train_samples.p', 'rb'))    
        elif train_or_test == 'test':
            output_name = 'viewsel_greedy_newcostmodel'
            samples = pickle.load(open(input_file_dir/'test_samples.p', 'rb'))
    else:
        all_samples = pickle.load(open(input_file_dir/'train_samples.p', 'rb'))
        k = int(sys.argv[1])
        split_num = int(sys.argv[2]) - 1
        if len(all_samples) % k != 0:
            print('Choose k that evenly divides # of samples')
            sys.exit()
        group_size = len(all_samples) / k
        begin_ind = int( split_num * group_size)
        end_ind = int( (split_num + 1) * group_size)
        # test samples are the smaller group, train samples are everything else
        if train_or_test == 'train':
            output_name = 'viewsel_train_greedy_newcostmodel'
            samples = copy.deepcopy(all_samples)
            del samples[begin_ind : end_ind]
        elif train_or_test == 'test':
            output_name = 'viewsel_greedy_newcostmodel'
            samples = all_samples[begin_ind : end_ind]
        suffix = '_k_'+str(k)+'_split_' + str(split_num + 1)
        output_name = output_name + suffix

    all_rewards = []
    output_fn = output_name+'.txt'
    file = open(input_file_dir/output_fn, 'w')
    file = open(input_file_dir/output_fn, 'a')

    env = ViewselEnv(input_file_dir, max_num_views, support_frac, suffix) 
    qry_to_views_fn = 'qry_to_views_max_'+str(max_num_views)+'_sf_'+str(support_frac)+'.p' 
    if os.path.exists(input_file_dir/qry_to_views_fn):
        qry_to_views = pickle.load(open(input_file_dir/qry_to_views_fn, 'rb'))
    else:
        qry_to_views = {} 
    # samples = [samples[1]]
    for num, input_queries in enumerate(samples):
        old_state = env.new_ep(input_queries, qry_to_views)
        old_state = np.reshape(old_state, (1, env.state.shape[0], 1))
        
        if tuple(input_queries) not in qry_to_views:
            qry_to_views[tuple(input_queries)] = env.get_views()

        if len(env.input_view_strs) == 0:  
            all_rewards.append(0)
            continue
        file.write('\nSample #: ' + str(num) + '\n')
        for q, qry in enumerate(input_queries):
            file.write('Input query #' + str(q) + ': ' + repr(qry) + ', ')
            file.write('Proc cost: ' + repr(env.qry_proc_costs[q])+ '\n')
        choice_num = 1
        while True:
            print('samp', num, 'step', choice_num)
            new_state, best_view, reward, done = env.step(file)  #run algo
            new_state = np.reshape(new_state, (1, env.state.shape[0], 1))
            memory.append([old_state, best_view, reward, new_state, done, [] ])
            old_state = new_state
            done = env.step(file)  #run algo
            if done:
                reward = env.reward
                break
            choice_num += 1
        all_rewards.append(round(reward, 4))

    pickle.dump( memory, open( input_file_dir/"greedy_memory"+suffix+".p", "wb" ) )
    # pickle.dump( qry_to_views, open( input_file_dir/qry_to_views_fn, "wb" ) )

    output_csvfn = output_name+'.csv'
    with open(input_file_dir/output_csvfn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['Support frac: ', support_frac, '', 'max num views: ', max_num_views])
        writer.writerow([])
        writer.writerow(['Samp num', 'Reward'])
        rew_sum=0
        for samp_num, rew in enumerate(all_rewards):
            writer.writerow([samp_num + begin_ind, rew])  #bc writerow works on lists, not ints
            rew_sum += rew
        writer.writerow(['Sum', rew_sum])  #bc writerow works on lists, not ints

if __name__ == "__main__":
  try:
    main()
  except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)