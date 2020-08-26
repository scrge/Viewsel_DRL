import pdb, pickle, random, os, copy
import psycopg2, pdb, pickle
from pathlib import Path

### USER PARAMETERS ###

Dataset = 'JOB_19'
num_train_queries = 33
num_test_queries = 0
num_train_samples = 50000  #number of sets of input queries in training set
num_test_samples = 0
min_num_queries_insamp = 25
max_num_queries_insamp = 30
# in_qry_threshold = 0.1  #+1 to view's "in qry count" if view is in X% of queries in workload
# overlap_threshold = 0.1  # % of views over total num views that are within 'in qry thrshold'
# max_num_views = 10
# support_frac = 0.25

###

output_path = Path.cwd() / Dataset
if not os.path.exists(output_path):
    os.makedirs(output_path)

conn = psycopg2.connect("dbname=imdbload user=postgres password=...")
cur = conn.cursor()

# table_sizes = {}
# cur.execute("""SELECT table_name FROM information_schema.tables
#        WHERE table_schema = 'public'""")
# for table in cur.fetchall():
#     print(table[0])
#     #cur.execute("SELECT * FROM " + table[0])
#     #print("The number of rows: ", cur.rowcount)
#     #table_sizes[table[0]] = cur.rowcount
#     cur.execute("select COUNT(*) from " + reln)
#     table_sizes[table[0]] = cur.fetchone()[0]
# pickle.dump( table_sizes, open( output_path+"named_reln_sizes.p", "wb" ) )

# get all attributes
#SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = N'title';

#################

folder = r"...\data\join_order_benchmark\queries" + "\\"
table_sizes = pickle.load(open(output_path/'named_reln_sizes.p', 'rb'))
sorted_table_names = sorted(table_sizes.keys(), key=lambda x:x.lower())

### Use this if want to use queries not in list of given queries for JOB benchmark
# schema = open(folder+'schema.sql', "r").readlines()
# reln_attr_dict = {table:[] for table in sorted_table_names}  #keys are relns, values are attr in those relns
# reln_attr_tuples = []
# for line in schema:
#     line = line.replace('\n','')
#     if 'CREATE TABLE' in line:
#         curr_reln = line.split()[2]
#     if 'CREATE TABLE' not in line and line != '' and ';' not in line:
#         reln_attr_dict[curr_reln].append(line.split()[0])
#         reln_attr_tuples.append(curr_reln + '.' + line.split()[0])

lst_queries = os.listdir(folder)
# lst_queries = [file for file in lst_queries if '.sql' in file and 'schema.sql' not in file and 'fkindexes' not in file]
# lst_queries = [qry for qry in lst_queries if 'a' in qry]
lst_queries = [str(i) + 'a.sql' for i in range(1,33)]

reln_attr_dict_qrys = {table:[] for table in sorted_table_names}  #keys are relns, values are attr in those relns
all_queries = []  #used to make queries
for qry_file in lst_queries:
    table_aliases = {}
    query_as_set = set()
    query = open(folder+qry_file, "r").readlines()
    from_flag = False
    for line in query:
        if 'FROM ' in line:
            from_flag = True
        if 'WHERE ' in line:
            from_flag = False
        if from_flag and ' AS ' in line:
            line = line.replace(',','')
            split_line = line.split()
            for i, mem in enumerate(split_line):
                if mem == 'AS' and split_line[i-1] in sorted_table_names:
                    table_aliases[split_line[i+1]] = split_line[i-1]
        if '=' in line:
            line = line.replace(';','')
            split_line = line.split()
            for i, mem in enumerate(split_line):
                if mem == '=' and '.' in split_line[i-1] and '.' in split_line[i+1]:
                    left = split_line[i-1].split('.')
                    left[0] = table_aliases[left[0]]
                    right = split_line[i+1].split('.')
                    right[0] = table_aliases[right[0]]
                    if left[0] == 'movie_link' or right[0] == 'movie_link':
                        continue  #movie_link size is 0 so ignore it
                    if left[0] == 'link_type' or right[0] == 'link_type':
                        continue  #link_type size is 0 so ignore it
                    join_cond = (".".join(left), ".".join(right))
                    if join_cond not in query_as_set:
                        query_as_set.add(frozenset(join_cond))
                    if left[1] not in reln_attr_dict_qrys[left[0]]:
                        reln_attr_dict_qrys[left[0]].append(left[1])
                    if right[1] not in reln_attr_dict_qrys[right[0]]:
                        reln_attr_dict_qrys[right[0]].append(right[1])
    all_queries.append(frozenset(query_as_set))

unique_all_queries = []
for qry in all_queries:
    if qry not in unique_all_queries:
        unique_all_queries.append(qry)
all_queries = unique_all_queries

del table_sizes['movie_link']
del reln_attr_dict_qrys['movie_link']
sorted_table_names.remove('movie_link')
del table_sizes['link_type']
del reln_attr_dict_qrys['link_type']
sorted_table_names.remove('link_type')

# Alphabetize attributes for within each relation
reln_attr_dict_qrys = {key:sorted(value) for key, value in reln_attr_dict_qrys.items()}

reln_attr_index = []  #index maps state matrix col index to (reln, attr)
first_ind_of_reln = {}
curr_pos = 0
for reln, attr_tuple in reln_attr_dict_qrys.items():
    first_ind_of_reln[reln] = curr_pos
    curr_pos += len(attr_tuple) #doesn't work if only 1 attr in reln!
    for attr in attr_tuple:
        reln_attr_index.append(reln+'.'+attr)

##############################################################################################
### Join selectivities
##############################################################################################

# How Good Are Query Optimizers, Really: join cardinality estimation:
# table_size(A) * table_size (B) / max ( distinct(x), distinct(y) )

if not os.path.exists(output_path/'distincts.p'):
    distincts = {}
    for reln_attr in reln_attr_index:
        reln_attr_tuple = reln_attr.split('.')
        reln = reln_attr_tuple[0]
        attr = reln_attr_tuple[1]
        print(reln, attr)
        cur.execute("select COUNT( DISTINCT " + attr + " ) from " + reln)
        distincts[(reln,attr)] = cur.fetchone()[0]
    pickle.dump( distincts, open( output_path+"distincts.p", "wb" ) )
else:
    distincts = pickle.load(open(output_path/'distincts.p', 'rb'))

join_selectivities = {}
for query in all_queries:
    query = tuple(query)
    for join_cond in query:
        join_cond = tuple(join_cond)
        distincts_1 = distincts[tuple(join_cond[0].split('.'))]
        distincts_2 = distincts[tuple(join_cond[1].split('.'))]
        if distincts_1 > 0 or distincts_2 > 0:
            jc_val = 1 / max(distincts_1, distincts_2)
            join_selectivities[join_cond] = jc_val   # tuple is hashable unlike list
            join_selectivities[(join_cond[1], join_cond[0])] = jc_val

##############################################################################################
### Convert to IDs
##############################################################################################

#keys are relns as number IDs, values are reln sizes
# reln_names_to_IDs = {}  #create and save for easy back and forth conversion; rather than using index()
# ID_table_sizes = []
# for i, reln in enumerate(reln_attr_dict_qrys.keys()):
#     reln_names_to_IDs[reln] = i + 1
#     #ID_table_sizes[i + 1] = table_sizes[reln]
#     ID_table_sizes.append(table_sizes[reln])
# table_sizes = ID_table_sizes

# convert relns and attrbs into IDs. each is an edge.
# lst_qry_ids = []  #index is qry ID
# for named_query in all_queries:
#     query_as_set = set()
#     named_query = tuple(named_query)
#     for join_cond in named_query:
#         left = join_cond[0].split('.')
#         left_reln = reln_names_to_IDs[left[0]]
#         left_attr = reln_attr_dict_qrys[left[0]].index(left[1])
#         right = join_cond[1].split('.')
#         right_reln = reln_names_to_IDs[right[0]]
#         right_attr = reln_attr_dict_qrys[right[0]].index(right[1])
#         edge = (str(left_reln) + '.' + str(left_attr), str(right_reln) + '.' + str(right_attr) )
#         query_as_set.add(frozenset(edge))   
#     lst_qry_ids.append(query_as_set)
# all_queries = lst_qry_ids

# reln_attr_index_IDs = []  
# for tup in reln_attr_index:
#     reln = reln_names_to_IDs[tup[0]]
#     attr = reln_attr_dict_qrys[tup[0]].index(tup[1])
#     reln_attr_index_IDs.append(str(reln)+'.'+str(attr))
# reln_attr_index = reln_attr_index_IDs

# convert joinsels to IDs here
# num_join_sels = {}
# for 

# pickle.dump( reln_names_to_IDs, open( output_path+"reln_names_to_IDs.p", "wb" ) )

##############################################################################################
### Save to files
##############################################################################################

pickle.dump( table_sizes, open( output_path/"reln_sizes.p", "wb" ) )
pickle.dump( reln_attr_dict_qrys, open( output_path/"reln_attr_dict_qrys.p", "wb" ) )
pickle.dump( all_queries, open( output_path/"all_queries.p", "wb" ) ) #113 queries
pickle.dump( reln_attr_index , open( output_path/"reln_attr_index.p", "wb" ) )
pickle.dump( join_selectivities, open(output_path/'join_selectivities.p', 'wb'))

#create freqs for every query
query_freqs = [10*random.randint(1,10) for q_ind in range(len(all_queries))]
pickle.dump( query_freqs, open( output_path/"query_freqs.p", "wb" ) )

# base_reln_update_freqs = {str(rel) : 10*random.randint(1,10) for rel in list(table_sizes.keys())}
base_reln_update_freqs = {str(rel) : 10*random.randint(1,3) for rel in list(table_sizes.keys())}
pickle.dump( base_reln_update_freqs, open( output_path/"base_reln_update_freqs.p", "wb" ) )

##############################################################################################
### Create workloads out of queries
##############################################################################################

# def sufficient_overlap(input_query_strs, all_queries, support, query_freqs, max_num_views,
#     in_qry_threshold, overlap_threshold, join_selectivities, reln_sizes):
#     lst_views = []
#     for q_ind, query in enumerate(input_query_strs):
#         query = list(query)
#         views_by_num_edges = {1:[]} # num_edges_in_view : views
#         # Each view is a frozenset of edges. Each edge is a frozenset
#         for edge in query:
#             freq_val = 0
#             for qry in input_query_strs:
#                 ind = all_queries.index(qry)
#                 if set([edge]).issubset(qry):
#                     freq_val += query_freqs[ind]
#             if freq_val > support:
#                 views_by_num_edges[1].append(set([edge]))  #view of 2 relns
#         edge_to_neighbors = {}  # edge : all edges that neighbor the key edge's RELATIONS
#         for edge_1 in query:
#             edge_1 = tuple(edge_1)
#             edge_1_relations = [edge_1[0].split('.')[0], edge_1[1].split('.')[0]]
#             for reln in edge_1_relations:
#                 for edge_2 in query:
#                     edge_2 = tuple(edge_2)
#                     edge_2_relations = [edge_2[0].split('.')[0], edge_2[1].split('.')[0]]
#                     if reln in edge_2_relations and edge_1 != edge_2:
#                         if frozenset(edge_1) in edge_to_neighbors:
#                             edge_to_neighbors[frozenset(edge_1)].append(frozenset(edge_2))
#                         else:
#                             edge_to_neighbors[frozenset(edge_1)] = [frozenset(edge_2)]
#         # add edges to new view
#         for num_edges in range(2, len(query)):
#             views_by_num_edges[num_edges] = []
#             prev_lvl = views_by_num_edges[num_edges - 1]
#             for lower_view in prev_lvl:
#                 lower_view_neighbors = []
#                 lower_view_lst = list(lower_view)
#                 for edge in lower_view_lst:
#                     lower_view_neighbors += edge_to_neighbors[edge]
#                 lower_view_neighbors = list(set(lower_view_neighbors))
#                 for neighbor in lower_view_neighbors:
#                     new_view = lower_view.union(set([neighbor])) 
#                     freq_val = 0   #check if new view freq > support threshold
#                     for qry in input_query_strs:
#                         ind = all_queries.index(qry)
#                         if set(new_view).issubset(qry):
#                             freq_val += query_freqs[ind]
#                     if freq_val > support and new_view not in views_by_num_edges[num_edges]:
#                         views_by_num_edges[num_edges].append(new_view)
#         for v_lst in views_by_num_edges.values():
#             lst_views.extend(v_lst)

#     new_lst_views = []
#     for view in lst_views:
#         if view not in new_lst_views:
#             new_lst_views.append(view)
#     lst_views = new_lst_views

#     lst_freq_vals = []
#     for view in lst_views:
#         freq_val = 0
#         for qry in input_query_strs:
#             ind = all_queries.index(qry)
#             if set(view).issubset(qry):
#                 freq_val += query_freqs[ind]
#         lst_freq_vals.append(freq_val)

#     view_sizes = []
#     for i, view in enumerate(lst_views):
#         view_size = 1
#         relns_so_far = []
#         for edge in view:
#             edge = tuple(edge)
#             left_reln = edge[0].split('.')[0]
#             right_reln = edge[1].split('.')[0]
#             join_sel = join_selectivities[edge]
#             view_size *= join_sel
#             if left_reln not in relns_so_far:
#                 view_size *= reln_sizes[left_reln]
#                 relns_so_far.append(left_reln)
#             if right_reln not in relns_so_far:
#                 view_size *= reln_sizes[right_reln]
#                 relns_so_far.append(right_reln)
#         view_sizes.append(view_size)

#     lst_tuples = list(zip(lst_views, lst_freq_vals, view_sizes))
#     sorted_lst_tuples = sorted(lst_tuples, key=lambda tup: (-tup[1], -tup[2])) 
#     sorted_lst_tuples.reverse()
#     sorted_lst_views = []
#     for tup in sorted_lst_tuples:
#         sorted_lst_views.append(tup[0])
#     top_views = sorted_lst_views[:max_num_views]  

#     ### Find # commons views for every query

#     #loop thru every view and check how many queries it belongs to
#     # +1 to 'in qry' counter each time view.issubset(qry)
#     top_views_qry_counter = [0] * len(top_views)
#     for v, view in enumerate(top_views):
#         for qry in input_query_strs:
#             if view.issubset(qry):
#                 top_views_qry_counter[v] += 1

#     # +1 to overlap counter if 'in qry' counter > threshold of % of workload
#     overlap_counter = 0
#     for count in top_views_qry_counter:
#         perc = count / len(input_query_strs)
#         if perc > in_qry_threshold:
#             overlap_counter += 1

#     #return True if overlap counter for % over total # of views > overlap threshold
#     return overlap_counter / len(top_views) > overlap_threshold

train_queries = all_queries[:num_train_queries]
test_queries = all_queries[num_train_queries:(num_train_queries+num_test_queries)]
samples = []
samples_as_sets = []
num_samples = num_train_samples + num_test_samples
while len(samples) < num_samples:
    print(len(samples), 'generating workload')
    num_input_qrys = random.randint(min_num_queries_insamp, max_num_queries_insamp)
    input_qry_set = []
    if len(samples) < num_train_samples:
        while len(input_qry_set) < num_input_qrys:
            input_q = random.choice(train_queries)
            if input_q not in input_qry_set:
                input_qry_set.append(input_q)
    else:
        while len(input_qry_set) < num_input_qrys:
            input_q = random.choice(test_queries)
            if input_q not in input_qry_set:
                input_qry_set.append(input_q) 
    # while len(input_qry_set) < num_input_qrys:
    #     input_q = random.choice(all_queries)  #randomly pick a query and put it in samp
    #     if input_q not in input_qry_set:
    #         input_qry_set.append(input_q) 

    #find # of overlapping views for each query
    # total_freq_val = 0
    # for qry in input_qry_set: 
    #     ind = all_queries.index(qry)
    #     total_freq_val += query_freqs[ind]
    # support = total_freq_val * support_frac
    # suff_overlap = sufficient_overlap(input_qry_set, all_queries, support, query_freqs, max_num_views, 
    #     in_qry_threshold, overlap_threshold, join_selectivities, table_sizes)
    # if input_qry_set not in samples and suff_overlap:
    
    ##Order workload by product of joinsels, largest to smallest (larger usually means less edges)
    prod_joinsel_lst = []
    for qry in input_qry_set:
        product_joinsel = 1
        for edge in qry:
            edge = tuple(edge)
            join_sel = join_selectivities[edge]
            product_joinsel *= join_sel
        prod_joinsel_lst.append(product_joinsel)
    lst_tuples = list(zip(input_qry_set, prod_joinsel_lst))
    sorted_lst_tuples = sorted(lst_tuples, key=lambda tup: (-tup[1])) 
    input_qry_set = [tup[0] for tup in sorted_lst_tuples]

    # if input_qry_set not in samples:
    if set(input_qry_set) not in samples_as_sets:
        samples.append(input_qry_set) 
        samples_as_sets.append(set(input_qry_set))  #keep as set bc need to check if view is subset of qry


train_samples = samples[:num_train_samples]
test_samples = samples[num_train_samples:]
pickle.dump(train_samples, open(output_path/'train_samples.p', 'wb'))
pickle.dump(test_samples, open(output_path/'test_samples.p', 'wb'))

# train_folder = output_path / 'train_samples'
# if not os.path.exists(train_folder):
#     os.makedirs(train_folder)

# for k, samp in enumerate(train_samples):
#     fn_num = 'train_' + str(k) + '.p'
#     pickle.dump( samp, open( train_folder/fn_num, "wb" ) )  
# for k, samp in enumerate(train_samples):
#     fn_num = 'train_' + str(k) + '.p'
#     pickle.dump( samp, open( train_folder/fn_num, "wb" ) )  

cur.close()


    