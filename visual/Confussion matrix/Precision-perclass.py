import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

Path_confusion_matrix = '../Confussion matrix/'
precision_file_SL4s_Ori = '../SME_train/SL_Animals_DVS_4s_Ori_8912/Precision.csv'
precision_file_SL4s_SME = '../SME_train/SL_Animals_DVS_4s_SME_9808/Precision.csv'
precision_file_SL4s_OF = '../SME_train/SL_Animals_DVS_4s_ERAFT_9179/Precision.csv'


# 開啟 CSV 檔案
with open(precision_file_SL4s_Ori, newline='') as csvfile:
    Precision_SL4s_Ori = np.array(list(csv.reader(csvfile)))

with open(precision_file_SL4s_SME, newline='') as csvfile:
    Precision_SL4s_SME = np.array(list(csv.reader(csvfile)))

with open(precision_file_SL4s_OF, newline='') as csvfile:
    Precision_SL4s_OF = np.array(list(csv.reader(csvfile)))
    

All_precision = np.zeros((19,4))
All_precision[:,0] = Precision_SL4s_Ori[:,0]
All_precision[:,1] = Precision_SL4s_Ori[:,1]
All_precision[:,2] = Precision_SL4s_SME[:,1]
All_precision[:,3] = Precision_SL4s_OF[:,1]
precision_data = All_precision.tolist()

df = pd.DataFrame(precision_data, columns=["class", "EVT", "MEVT(VME)", "MEVT(OF)"])
df['class'] = df['class'].astype('string').str[:-2]
df.plot(x="class", y=["EVT", "MEVT(VME)", "MEVT(OF)"], kind="bar", figsize=(9, 6), color=['gray', 'yellow', 'blue'], edgecolor = 'black')
plt.xticks(rotation=0, horizontalalignment="center")


y_major_locator=MultipleLocator(0.1)
plt.gca().yaxis.set_major_locator(y_major_locator)
plt.ylim(0,1,0.1)

plt.legend(loc="lower right")
plt.ylabel('Precision', color='black')
plt.xlabel('Class', color='black')

plt.grid(color='gray', axis='y', linestyle = '--')

fig_name = 'Per-class precision'
plt.title(fig_name)
fig = plt.gcf()
plt.show()
fig.savefig(Path_confusion_matrix + '/' + fig_name+'.TIFF',dpi=600, bbox_inches='tight')
