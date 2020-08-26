import csv, pdb, pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dataset_name = 'JOB_19'
fn = 'norewrites_run_1_k_10_split_1'
compare_to = 'opt'
train_intv = 25000
intv = 10000
# train_frac_opt = pickle.load(open(fn+'_train_frac_'+compare_to+'.p', 'rb'))
frac_opt = pickle.load(open(fn+'_test_frac_'+compare_to+'.p', 'rb'))
# train_indivdiff = pickle.load(open(fn+'_train_indivdiff_avgs.p', 'rb'))
test_indivdiff = pickle.load(open(fn+'_test_indivdiff_cols.p', 'rb'))[-1]
all_test_fracopt = pickle.load(open(fn+'_all_test_fracopt.p', 'rb'))
transpose_all_fracopt = np.array(all_test_fracopt).T.tolist()

iters = pickle.load(open(fn+'_iters_per_ep.p', 'rb'))
eps = [x*intv for x in range(len(frac_opt))]
eps[0] = 1
iters_2 = []
eps_2 = []
for i, num in enumerate(iters):
    # if i % 2 == 0:
    if eps[i] % 20000 != 0:
        iters_2.append('')
        eps_2.append('')
    else:
        iters_2.append(num)
        eps_2.append(eps[i])
iters = iters_2
# eps = eps_2

# x_labels = [str((p,q)) for p,q in zip(eps, iters)]
x_labels = [str(p) +'\n' +str(q) for p,q in zip(eps_2, iters)]
# train_eps = [x*train_intv for x in range(len(train_indivdiff))]
# train_eps[0] = 1

# rows = []
# input_file_dir = Path.cwd().parents[1] / "datasets" / dataset_name 
# opt_fn = 'viewsel_greedy_newcostmodel_k_10_split_1.csv'
# file = open(input_file_dir/opt_fn, 'r')
# read_file = csv.reader(file)
# for line in read_file:
#     if line:
#         rows.append(line)
# greedy_val = float(rows[-1][-1]) 

# rows = []
# input_file_dir = Path.cwd().parents[1] / "datasets" / dataset_name 
# opt_fn = 'viewsel_iterimprv_newcostmodel_k_10_split_1.csv'
# file = open(input_file_dir/opt_fn, 'r')
# read_file = csv.reader(file)
# for line in read_file:
#     if line:
#         rows.append(line)
# iter_imprv_val = float(rows[-1][-1])

# iter_imprv_frac = iter_imprv_val / greedy_val

# if train_frac_opt:
#     fig = plt.figure()
#     plt.plot(train_eps, [1.0 for x in range(len(train_frac_opt))], 'r-', label = 'Optimal')
#     plt.plot(train_eps, [0.9 for x in range(len(train_frac_opt))], marker='.', color='lightgrey')
#     plt.plot(train_eps, [0.8 for x in range(len(train_frac_opt))], marker='.', color='lightgrey')
#     # plt.plot([train_eps, [iter_imprv_frac for x in range(len(train_frac_opt))], 'y-', label = 'iter_imprv')
#     plt.plot(train_eps, train_frac_opt, marker='o', color='blue', label='test')
#     plt.xlabel('Episodes', fontsize=16)
#     # plt.ylabel('Frac Regret', fontsize=16)
#     plt.title('Train Frac Regret', fontsize=16)
#     plt.legend(prop={'size': 15}, loc='lower right')
#     fig.savefig(fn+'_train_frac_eps.png')
#     plt.show()
#     plt.close()

# fig = plt.figure()
# plt.plot(eps, [1.0 for x in range(len(frac_opt))], 'r-', label = 'greedy')
# plt.plot(eps, [0.9 for x in range(len(frac_opt))], marker='.', color='lightgrey')
# plt.plot(eps, [0.8 for x in range(len(frac_opt))], marker='.', color='lightgrey')
# plt.plot(eps, [iter_imprv_frac for x in range(len(frac_opt))], 'y-', label = 'iter_imprv')
# plt.plot(eps, transpose_all_fracopt[-1], marker='o', color='blue', label='test')
# plt.xlabel('Episodes', fontsize=16)
# plt.title('Samp #999 Test Frac Regret', fontsize=16)
# # plt.ylabel('Frac Regret', fontsize=16)
# plt.legend(prop={'size': 15}, loc='lower right')
# fig.savefig(fn+'_test_frac_eps_samp999.png')
# plt.show()
# plt.close()

fig = plt.figure()
plt.plot(eps, [1.0 for x in range(len(frac_opt))], 'r-', label = 'Optimal')
plt.plot(eps, [0.9 for x in range(len(frac_opt))], marker='.', color='lightgrey')
plt.plot(eps, [0.97 for x in range(len(frac_opt))], marker='.', color='lightgrey')
# plt.plot(eps, [iter_imprv_frac for x in range(len(frac_opt))], 'y-', label = 'iter_imprv')
plt.plot(eps, frac_opt, marker='o', color='blue', label='test')
plt.xlabel('Episodes', fontsize=16)
plt.title('Test Frac Regret', fontsize=16)
# plt.ylabel('Frac Regret', fontsize=16)
plt.legend(prop={'size': 15}, loc='lower right')
fig.savefig(fn+'_test_frac_eps.png')
plt.show()
plt.close()

# if train_indivdiff:
#     fig = plt.figure()
#     plt.plot(train_eps, [1.0 for x in range(len(train_indivdiff))], 'r-', label = 'Optimal')
#     plt.plot(train_eps, [0.9 for x in range(len(train_indivdiff))], marker='.', color='lightgrey')
#     plt.plot(train_eps, [0.8 for x in range(len(train_indivdiff))], marker='.', color='lightgrey')
#     # plt.plot(train_eps, [iter_imprv_frac for x in range(len(train_indivdiff))], 'y-', label = 'iter_imprv')
#     plt.plot(train_eps, train_indivdiff, marker='o', color='blue', label='test')
#     plt.xlabel('Episodes', fontsize=16)
#     # plt.ylabel('Avg of indiv frac regret', fontsize=16)
#     plt.title('Train Avg of Indiv Frac Regret', fontsize=16)
#     plt.legend(prop={'size': 15}, loc='lower right')
#     fig.savefig(fn+'_train_indiv_diff_eps.png')
#     plt.show()
#     plt.close()

# if test_indivdiff:
#     fig = plt.figure()
#     plt.plot(eps, [1.0 for x in range(len(frac_opt))], 'r-', label = 'Optimal')
#     plt.plot(eps, [0.9 for x in range(len(frac_opt))], marker='.', color='lightgrey')
#     plt.plot(eps, [0.8 for x in range(len(frac_opt))], marker='.', color='lightgrey')
#     # plt.plot(eps, [iter_imprv_frac for x in range(len(frac_opt))], 'y-', label = 'iter_imprv')
#     plt.plot(eps, test_indivdiff, marker='o', color='blue', label='test')
#     plt.xlabel('Episodes', fontsize=16)
#     # plt.ylabel('Avg of indiv frac regret', fontsize=16)
#     plt.title('Test Avg of Indiv Frac Regret', fontsize=16)
#     plt.legend(prop={'size': 15}, loc='lower right')
#     fig.savefig(fn+'_test_indiv_diff_eps.png')
#     plt.show()
#     plt.close()

# fig = plt.figure()
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.plot(eps, [1.0 for x in range(len(frac_opt))], 'r-', label = 'Optimal')
# plt.plot(eps, [0.9 for x in range(len(frac_opt))], marker='.', color='lightgrey')
# plt.plot(eps, [0.8 for x in range(len(frac_opt))], marker='.', color='lightgrey')
# # plt.plot(x_labels, [iter_imprv_frac for x in range(len(frac_opt))], 'y-', label = 'iter_imprv')
# plt.plot(eps, frac_opt, marker='o', color='blue', label='test')
# plt.xlabel('Eps / Iters', fontsize=16)
# # plt.ylabel('Frac Regret', fontsize=16)
# plt.title('Test Frac Regret', fontsize=16)
# plt.xticks(eps, x_labels)
# # plt.xticks(eps, iters)
# plt.rc('xtick',labelsize=0.75)
# plt.rc('ytick',labelsize=1)
# plt.legend(prop={'size': 15}, loc='lower right')
# fig.savefig(fn+'_test_frac_eps_iters.png')
# plt.show()
# plt.close()