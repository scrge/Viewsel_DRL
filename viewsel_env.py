#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math, random, pickle, itertools, copy, re, os.path, time
from functools import reduce
from pathlib import Path
import pdb, traceback, sys
import numpy as np

class ViewselEnv():
    def __init__(self, input_file_dir, max_num_queries, max_num_views, support_frac, suffix):
        #The following are all fixed given DB (same in every episode)
        self.all_queries = pickle.load(open(input_file_dir/'all_queries.p', 'rb'))  #all possible queries
        self.all_query_freqs = pickle.load(open(input_file_dir/'query_freqs.p', 'rb'))

        #set reln sizes + joinsel outside of making env; should be the same each time. Specific to each DB.
        self.reln_sizes = pickle.load(open(input_file_dir/'reln_sizes.p', 'rb'))
        self.join_selectivities = pickle.load(open(input_file_dir/'join_selectivities.p', 'rb'))
        self.reln_attr_index = pickle.load(open(input_file_dir/'reln_attr_index.p', 'rb'))  #keys are relations. values are which attributes those relations contain

        self.max_num_queries = max_num_queries
        self.max_num_views = max_num_views
        self.support_frac = support_frac
        self.RA = len(self.reln_attr_index)  #should be specific DB to DB

        qry_to_views_fn = 'qry_to_views_max_'+str(max_num_views)+'_sf_'+str(support_frac)+'.p' 
        self.qry_to_views_path = input_file_dir/qry_to_views_fn
        if os.path.exists(self.qry_to_views_path):
            self.qry_to_views = pickle.load(open(self.qry_to_views_path, 'rb'))
        else:
            self.qry_to_views = {}

        if os.path.exists(input_file_dir/'base_reln_update_freqs.p'):
            self.base_reln_update_freqs = pickle.load(open(input_file_dir/'base_reln_update_freqs.p', 'rb'))  
        else:
            self.base_reln_update_freqs = [1 for i in range(len(self.reln_sizes))]

        #dummy used to define state tensor shape when init DQN
        self.input_queries = np.zeros((self.max_num_queries, self.RA, self.RA))
        self.input_views = np.zeros((self.max_num_views, self.RA, self.RA))
        self.views_selected = np.zeros(self.max_num_views)
        #self.views_sub_which_queries = np.zeros((self.max_num_queries, self.max_num_views))
        self.query_rewrites = np.zeros((self.max_num_queries, self.max_num_views))
        self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), self.views_selected, self.query_rewrites.flatten()), axis=0)
        # self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), 
        #     self.views_sub_which_queries.flatten(), self.views_selected), axis=0)

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

    def new_ep(self, input_query_strs):    #Reset the state of the environment with new inputs
        #input query reps, viewsel, query rewrites, state
        self.input_queries = np.zeros((self.max_num_queries, self.RA, self.RA))
        self.input_views = np.zeros((self.max_num_views, self.RA, self.RA))
        self.views_selected = np.zeros(self.max_num_views)
        self.query_rewrites = np.zeros((self.max_num_queries, self.max_num_views))

        #At start of an episode, get input queries. Put them into state matrix.
        self.input_query_strs = list(input_query_strs)
        self.num_input_qrys = len(self.input_query_strs)

        if tuple(input_query_strs) in self.qry_to_views:
            view_tuple = self.qry_to_views[tuple(input_query_strs)]
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
                self.qry_to_views[tuple(input_query_strs)] = \
                    (self.input_view_strs, self.view_sizes, self.view_costs, \
                        self.relns_of_view, self.relns_of_input_queries, self.qry_freqs)
                # pickle.dump( self.qry_to_views, open( self.qry_to_views_path, "wb" ) )

        # self.views_sub_which_queries = np.zeros((self.max_num_queries, self.max_num_views))
        # for q_ind, qry in enumerate(self.input_query_strs):
        #     for v_ind, view in enumerate(self.input_view_strs):
        #         if view.issubset(qry):
        #             self.views_sub_which_queries[q_ind][v_ind] = 1

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
        # self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), 
        #     self.views_sub_which_queries.flatten(), self.views_selected), axis=0)
        self.state = self.state.astype(int)

        # Store what the agent tried
        self.lst_relns_used_in_rewriting = [[]]*self.num_input_qrys
        self.views_tried = [] #list of actions mapped to views. modified in train_...py
        self.done = False
        self.qry_proc_costs = {} #q: proc cost of curr rewrite
        self.maint_cost_lst = {} #view ID : maintcost, for selected views
        self.strikes = 0
        self.strike_limit = 1
        self.prev_best_state = self.state[:]
        self.prev_best_maint_cost_lst = {}
        self.prev_best_views_tried = []
        self.prev_best_views_selected = copy.deepcopy(self.views_selected)
        self.prev_best_query_rewrites = copy.deepcopy(self.query_rewrites)
        self.init_cost, self.prev_best_qry_costs = self._get_eval_cost(None, 'all', [])
        self.prev_best_reward = 0
        self.reward = 0
        self.prev_best_maint_cost = 0
        self.prev_best_eval_cost = self.init_cost
        self.prev_best_new_cost = self.init_cost
        return self.state

    def step(self, action):
        if action == -1:
            self.state = copy.deepcopy(self.prev_best_state)
            self.reward = self.prev_best_reward
            self.maint_cost_lst = copy.deepcopy(self.prev_best_maint_cost_lst)
            self.qry_proc_costs = copy.deepcopy(self.prev_best_qry_costs)
            self.views_selected = copy.deepcopy(self.prev_best_views_selected)
            self.query_rewrites = copy.deepcopy(self.prev_best_query_rewrites)
            self.done = True
            return self.state, self.reward, self.done

        self.views_selected[action] = 1  #write selview into state tensor

        #choose rewritings
        for q_ind, qry in enumerate(self.input_query_strs):
            if self.input_view_strs[action].issubset(qry):
                no_sub_or_super = True
                subviews_of_action = []
                for sel_view, v_bool in enumerate(self.query_rewrites[q_ind]):
                    if v_bool == 1: #view used in rewriting of query q_ind
                        is_sub = self.input_view_strs[action].issubset(self.input_view_strs[sel_view]) #v3
                        is_super = self.input_view_strs[sel_view].issubset(self.input_view_strs[action]) #v3
                        # if is_sub and is_super:
                        if is_sub:
                            no_sub_or_super = False  #don't add view if it's subview of existing view in rewriting
                        if is_super:
                            subviews_of_action.append(sel_view)
                if no_sub_or_super: #dont add if new view is subview of existing views in rewriting
                    new_eval_cost = self._get_eval_cost(action, q_ind, subviews_of_action)
                    if self.prev_best_qry_costs[q_ind] > new_eval_cost:
                        for v_ind in subviews_of_action: #if new view is superview of existing views in rewriting, must remove them to add
                            self.query_rewrites[q_ind][v_ind] = 0 #replace subviews in rewriting
                        self.query_rewrites[q_ind][action] = 1 #add superview to rewriting

        #remove any views not used in rewriting
        #for each view, check if it's in at least on rewriting. if not, remove it.
        for v_ind, v_bool in enumerate(self.views_selected):
            if v_bool == 1:
                v_col = self.query_rewrites[:,v_ind]
                if 1 not in v_col:
                    self.views_selected[v_ind] = 0

        self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), self.views_selected, self.query_rewrites.flatten()), axis=0)
        # self.state = np.concatenate((self.input_queries.flatten(), self.input_views.flatten(), 
        #             self.views_sub_which_queries.flatten(), self.views_selected), axis=0)
        self.state = self.state.astype(int)

        self.strikes += 1
        reward = self._get_reward(action)
        if reward > self.prev_best_reward:  #replace and start strike count over
            self.prev_best_state = copy.deepcopy(self.state)
            self.prev_best_reward = reward
            self.prev_best_maint_cost_lst = copy.deepcopy(self.maint_cost_lst)
            self.prev_best_qry_costs = copy.deepcopy(self.qry_proc_costs)
            self.prev_best_views_tried = copy.deepcopy(self.views_tried)
            self.prev_best_views_selected = copy.deepcopy(self.views_selected)
            self.prev_best_query_rewrites = copy.deepcopy(self.query_rewrites)
            self.prev_best_maint_cost = copy.deepcopy(self.maint_cost)
            self.prev_best_eval_cost = copy.deepcopy(self.eval_cost)
            self.prev_best_new_cost = copy.deepcopy(self.new_cost)
            self.strikes = 0
        elif self.strikes >= self.strike_limit:  #revert back to prev best and end
            self.state = copy.deepcopy(self.prev_best_state)
            self.reward = self.prev_best_reward
            self.maint_cost_lst = copy.deepcopy(self.prev_best_maint_cost_lst)
            self.qry_proc_costs = copy.deepcopy(self.prev_best_qry_costs)
            self.views_tried = copy.deepcopy(self.prev_best_views_tried)
            self.views_selected = copy.deepcopy(self.prev_best_views_selected)
            self.query_rewrites = copy.deepcopy(self.prev_best_query_rewrites)
            self.maint_cost = self.prev_best_maint_cost
            self.eval_cost = self.prev_best_eval_cost
            self.new_cost = self.prev_best_new_cost
            self.done = True

        if len(self.views_tried) == self.max_num_views:
            self.done = True
        return self.state, self.reward, self.done

    def _get_eval_cost(self, new_view, qry_num, subviews_of_action):
        processing_cost = 0 #cost of repr all queries
        qry_proc_costs = {}
        temp_relns_in_rewriting = copy.deepcopy(self.lst_relns_used_in_rewriting)
        if qry_num == 'all':
            qry_lst = range(self.num_input_qrys)
            if new_view != None:
                for q_ind in qry_lst:
                    if self.query_rewrites[q_ind][new_view] == 1:
                        temp_relns_in_rewriting[q_ind] = list(set(temp_relns_in_rewriting[q_ind] + \
                            self.relns_of_view[new_view]))
        else:
            qry_lst = [qry_num]
            temp_relns_in_rewriting[qry_num] = list(set(temp_relns_in_rewriting[qry_num] + \
                self.relns_of_view[new_view]))

        for q_ind in qry_lst:
            rewrite_lst_views_relns = []
            rewrite_lst_view_sizes = [] #sizes of views in this query's rewrite
            if qry_num != 'all':
                rewrite_lst_view_sizes.append(self.view_sizes[new_view])
                rewrite_lst_views_relns.append(self.relns_of_view[new_view])
            for i, v in enumerate(self.query_rewrites[q_ind]):
                if i in subviews_of_action:
                    continue
                if v == 1:
                    rewrite_lst_view_sizes.append(self.view_sizes[i])
                    rewrite_lst_views_relns.append(self.relns_of_view[i])
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
            self.lst_relns_used_in_rewriting = copy.deepcopy(temp_relns_in_rewriting)
            return processing_cost, qry_proc_costs
        else:
            return processing_cost

    def _get_reward(self, new_view):
        #total maint cost: add up all maint costs of views
        #maint cost of view: proccost of view 
        self.maint_cost_lst = {}
        maint_cost = 0
        for i, v in enumerate(self.views_selected[:len(self.input_view_strs)]): #get views selected
            if v == 1: #view is selected
                maint_cost += self.view_costs[i]
                self.maint_cost_lst[i] = self.view_costs[i]
        eval_cost, self.qry_proc_costs = self._get_eval_cost(new_view, 'all', [])
        self.maint_cost = maint_cost
        self.eval_cost = eval_cost
        self.new_cost = (maint_cost + eval_cost)
        if self.init_cost - self.new_cost > 0:
            self.reward = (self.init_cost - self.new_cost)/(self.init_cost)
        else:
            self.reward = -1
        return self.reward
        
    def seed(self, seed):
        random.seed(seed)
        np.random.seed