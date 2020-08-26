import csv, pdb, pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import PercentFormatter

dataset_name = 'JOB_15'
fn = 'norewrites_run_1_k_10_split_1'
num_samp = 1000 #test samps total
compare_to = 'opt'
all_histos = pickle.load(open(fn+'_rew_histos.p', 'rb'))  #list of lists. each list is a histo
transpose_histos = np.array(all_histos).T.tolist()
intv = 5000
eps = [x*intv for x in range(len(transpose_histos[-1]))]
# pdb.set_trace()

# cats = [str(i*10)+'-'+str(i*10 +9)+'%' for i in range(10)] + ['100%', '> 100%']
cats = [str(i*10)+'%' for i in range(10)] + ['100%', '>100']

fig = plt.figure()
freqs = all_histos[0]
plt.bar(cats, freqs)
plt.xlabel("Freq", fontsize=16)
plt.ylabel("Frac Regret Interval", fontsize=16)
plt.title("Distr. of Frac Regret Values, Ep"+str(eps[0]), fontsize=16)
x_pos = [i for i, _ in enumerate(cats)]
plt.xticks(x_pos, cats)
fig.savefig('histo_ep0.png')
plt.show()
plt.close()

fig = plt.figure()
freqs = all_histos[-1]
plt.bar(cats, freqs)
plt.xlabel("Freq", fontsize=16)
plt.ylabel("Frac Regret Interval", fontsize=16)
plt.title("Distr. of Frac Regret Values, Ep"+str(100000), fontsize=16)
x_pos = [i for i, _ in enumerate(cats)]
plt.xticks(x_pos, cats)
fig.savefig('histo_ep100000.png')
plt.show()
plt.close()

# fig = plt.figure()
# y = transpose_histos[0]
# y = [(i/num_samp)*100 for i in y]
# plt.plot(eps, y, 'r-')
# # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.xlabel('Episodes', fontsize=16)
# plt.title('Percent of samples < 10%', fontsize=16)
# fig.savefig('less_than_10.png')
# plt.show()
# plt.close()

fig = plt.figure()
y = transpose_histos[0]
for j in range(1,7):
    y = [sum(x) for x in zip(y, transpose_histos[j])]
y = [(i/num_samp)*100 for i in y]
plt.plot(eps, y, 'r-')
plt.xlabel('Episodes', fontsize=16)
plt.title('Percent of samples < 70%', fontsize=16)
fig.savefig('less_than_70.png')
plt.show()
plt.close()


fig = plt.figure()
y = transpose_histos[9]
for j in range(10,12):
    y = [sum(x) for x in zip(y, transpose_histos[j])]
y = [(i/num_samp)*100 for i in y]
plt.plot(eps, y, 'r-')
plt.xlabel('Episodes', fontsize=16)
plt.title('Percent of samples >= 90%', fontsize=16)
fig.savefig('more_than_90.png')
plt.show()
plt.close()

# fig = plt.figure()
# y = transpose_histos[-3]
# y = [(i/num_samp)*100 for i in y]
# plt.plot(eps, y, 'r-')
# plt.xlabel('Episodes', fontsize=16)
# plt.title('Percent of samples b/w 90 to 99%', fontsize=16)
# fig.savefig('90_to_99.png')
# plt.show()
# plt.close()

# fig = plt.figure()
# y = transpose_histos[-2]
# y = [(i/num_samp)*100 for i in y]
# plt.plot(eps, y, 'r-')
# plt.xlabel('Episodes', fontsize=16)
# plt.title('Percent of samples = 100%', fontsize=16)
# fig.savefig('same_as_100.png')
# plt.show()
# plt.close()

# #plot the progress of category >100%
# fig = plt.figure()
# y = transpose_histos[-1]
# y = [(i/num_samp)*100 for i in y]
# plt.plot(eps, y, 'r-')
# plt.xlabel('Episodes', fontsize=16)
# plt.title('Percent of samples > 100%', fontsize=16)
# fig.savefig('greater_than_100.png')
# plt.show()
# plt.close()