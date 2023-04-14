import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

file_name = './Frequency/cifar10_std_class0.npy'

res = np.load(file_name)

res = res.reshape(2, 196)
statis_results_robust = res[0]
statis_results_std = res[1]

save_name = 'cifar10_ViT_std_clip_1e-2'

fig = plt.figure(figsize=(10, 10))
predict_res_1 = statis_results_std
predict_res_2 = statis_results_robust
index = np.argsort(-predict_res_1)
# index = np.argsort(-predict_res_2)
predict_res_1 = predict_res_1[index]
predict_res_2 = predict_res_2[index]


left, bottom, width, height = 0.2, 0.15, 0.78, 0.68
ax1 = fig.add_axes([left, bottom, width, height])
ax1.set_ylim(0, 1000)

ax1.bar(range(int(len(predict_res_1))), predict_res_1[range(0, len(predict_res_1))], color='royalblue', label='Samples without semantic noise')
ax1.bar(range(int(len(predict_res_1))), predict_res_2[range(0, len(predict_res_1))], color='indianred', alpha=0.7, label='Samples with semantic noise')

ax1.legend(loc=0, ncol=1, prop={'family': 'Times New Roman', 'size': 16})
ax1.grid(linestyle='--', color='lightgray')
ax1.set_ylabel('Number of activation', fontdict={'size': 16})
ax1.set_xlabel('Feature index', fontdict={'size': 16})
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=14)


if os.path.exists('./Figure') == False:
    os.makedirs('./Figure')
plt.savefig('./Figure/%s.pdf' % (save_name))
plt.savefig('./Figure/%s.png' % (save_name))