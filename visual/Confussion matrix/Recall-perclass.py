import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

Path_confusion_matrix = '../Confussion matrix/'
recall_file_SL4s_Ori = '../SME_train/SL_Animals_DVS_4s_Ori_8912/Recall.csv'
recall_file_SL4s_SME = '../SME_train/SL_Animals_DVS_4s_SME_9808/Recall.csv'
recall_file_SL4s_OF = '../SME_train/SL_Animals_DVS_4s_ERAFT_9179/Recall.csv'


# 開啟 CSV 檔案
with open(recall_file_SL4s_Ori, newline='') as csvfile:
    Recall_SL4s_Ori = np.array(list(csv.reader(csvfile)))

with open(recall_file_SL4s_SME, newline='') as csvfile:
    Recall_SL4s_SME = np.array(list(csv.reader(csvfile)))

with open(recall_file_SL4s_OF, newline='') as csvfile:
    Recall_SL4s_OF = np.array(list(csv.reader(csvfile)))
    

All_recall = np.zeros((19,4))
All_recall[:,0] = Recall_SL4s_Ori[:,0]
All_recall[:,1] = Recall_SL4s_Ori[:,1]
All_recall[:,2] = Recall_SL4s_SME[:,1]
All_recall[:,3] = Recall_SL4s_OF[:,1]
recall_data = All_recall.tolist()

df = pd.DataFrame(recall_data, columns=["class", "EVT", "MEVT(VME)", "MEVT(OF)"])
df['class'] = df['class'].astype('string').str[:-2]
df.plot(x="class", y=["EVT", "MEVT(VME)", "MEVT(OF)"], kind="bar", figsize=(9, 6), color=['gray', 'yellow', 'blue'], edgecolor = 'black')
plt.xticks(rotation=0, horizontalalignment="center")

y_major_locator=MultipleLocator(0.1)
plt.gca().yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1)

plt.legend(loc="lower right")
plt.ylabel('Recall', color='black')
plt.xlabel('Class', color='black')

plt.grid(color='gray', axis='y', linestyle = '--')

fig_name = 'Per-class recall'
plt.title(fig_name)
fig = plt.gcf()
plt.show()
fig.savefig(Path_confusion_matrix + '/' + fig_name+'.TIFF',dpi=600, bbox_inches='tight')
