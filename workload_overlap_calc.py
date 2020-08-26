import pickle, pdb, math
from pathlib import Path

dataset_name = 'JOB_6_2'
input_file_dir = Path.cwd().parent / "datasets" / dataset_name 
samples = pickle.load(open(input_file_dir/'train_samples.p', 'rb'))    
test_samples = pickle.load(open(input_file_dir/'test_samples.p', 'rb'))

workload_overlap_vals = []
for workload_lst in samples:
    #loop thru all queries, recording edges and +1 for each qry the edge appears in
    edge_freq = {} #frozenset(edge) : # queries the edge appears in
    for qry in workload_lst:
        for edge in qry:
            if edge not in edge_freq:
                edge_freq[edge] = 1
            else:
                edge_freq[edge] += 1

    #Average for all edges: 
    #1) Add up all edge_fracs to get edge_frac_sum
    #2) Divide to get edge_frac_sum / (total # edges)
    #Alternatively, it looks like: (edge_sum) / (total # edges * total # queries)
    #For each edge: edge frac = (# queries that share that edge) / (total # queries)

    total = sum(edge_freq.values())
    overlap_val = total / ( len(edge_freq) * len(workload_lst) )
    workload_overlap_vals.append(overlap_val)

samp_result_histo = [0] * 11
for val in workload_overlap_vals:
    if val < 1:
        samp_result_histo[math.floor(val*10)] += 1
    elif val == 1:
        samp_result_histo[10] += 1

print(samp_result_histo)
pdb.set_trace()

#use this to make sure, when generating workloads, each workload is above a certain overlap threshold

