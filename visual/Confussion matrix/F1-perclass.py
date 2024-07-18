import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

Path_confusion_matrix = '../Confussion matrix/'
f1_file_SL4s_Ori = '../SME_train/SL_Animals_DVS_4s_Ori_8912/F1.csv'
f1_file_SL4s_SME = '../SME_train/SL_Animals_DVS_4s_SME_9808/F1.csv'
f1_file_SL4s_OF = '../SME_train/SL_Animals_DVS_4s_ERAFT_9179/F1.csv'


# 開啟 CSV 檔案
with open(f1_file_SL4s_Ori, newline='') as csvfile:
    F1_SL4s_Ori = np.array(list(csv.reader(csvfile)))

with open(f1_file_SL4s_SME, newline='') as csvfile:
    F1_SL4s_SME = np.array(list(csv.reader(csvfile)))

with open(f1_file_SL4s_OF, newline='') as csvfile:
    F1_SL4s_OF = np.array(list(csv.reader(csvfile)))
    

All_f1 = np.zeros((19,4))
All_f1[:,0] = F1_SL4s_Ori[:,0]
All_f1[:,1] = F1_SL4s_Ori[:,1]
All_f1[:,2] = F1_SL4s_SME[:,1]
All_f1[:,3] = F1_SL4s_OF[:,1]
f1_data = All_f1.tolist()

df = pd.DataFrame(f1_data, columns=["class", "EVT", "MEVT(VME)", "MEVT(OF)"])
df['class'] = df['class'].astype('string').str[:-2]
df.plot(x="class", y=["EVT", "MEVT(VME)", "MEVT(OF)"], kind="bar", figsize=(9, 6), color=['gray', 'yellow', 'blue'], edgecolor = 'black')
plt.xticks(rotation=0, horizontalalignment="center")

y_major_locator=MultipleLocator(0.1)
plt.gca().yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)

plt.legend(loc="lower right")
plt.ylabel('F1', color='black')
plt.xlabel('Class', color='black')

plt.grid(color='gray', axis='y', linestyle = '--')

fig_name = 'Per-class f1-score'
plt.title(fig_name)
fig = plt.gcf()
plt.show()
fig.savefig(Path_confusion_matrix + '/' + fig_name+'.TIFF',dpi=600, bbox_inches='tight')
