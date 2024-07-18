import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

Path_confusion_matrix = '../Confussion matrix/'
Methods = ["EVT", "MEVT(VME)", "MEVT(OF)"]
x = np.arange(len(Methods))
micro_f1_data = [0.8702, 0.9181, 0.8807]  

plt.bar(x, micro_f1_data, alpha=0.9, width = 0.35, edgecolor = 'black', lw=1, color=['gray', 'yellow', 'blue'])

plt.xticks(x, Methods)      
plt.ylim(0,1.1)
plt.ylabel('F1', color='black')

fig_name = 'Micro_f1'
plt.title(fig_name)
fig = plt.gcf()
plt.show()
fig.savefig(Path_confusion_matrix + '/' + fig_name+'.TIFF',dpi=600, bbox_inches='tight')
