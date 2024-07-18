import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSVFILE = './pretrained_models/tests/0605_1722_model_285/train_log/version_0/metrics.csv'
df = pd.read_csv(CSVFILE)
epoch = np.zeros(190)

train_acc = np.zeros(190)
test_acc = np.zeros(190)
train_loss = np.zeros(190)
test_loss = np.zeros(190)

for n in range(190):
    epoch[n] = df['epoch'][3*n+1]
    train_acc[n] = df['train_acc'][3*n+2]
    test_acc[n] = df['val_acc'][3*n+1]
    train_loss[n] = df['train_loss_total'][3*n+2]
    test_loss[n] = df['val_loss_total'][3*n+1]
    
train_acc_max = np.argmax(train_acc)
test_acc_max = np.argmax(test_acc)
train_loss_min = np.argmin(train_loss)
test_loss_min = np.argmin(test_loss)

breakpoint()


# Train Acc.
plt.xlabel('epoch')
plt.ylabel('train_acc')
plt.title('')
plt.plot(train_acc)
plt.annotate('(%.4f)'%(train_acc[train_acc_max]), xy = (train_acc_max, train_acc[train_acc_max]))
plt.show()



# Train Loss 
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.title('')
plt.plot(train_loss)
plt.annotate('(%.4f)'%train_loss[train_loss_min], xy = (train_loss_min, train_loss[train_loss_min]))
plt.show()



# Test Acc.
plt.xlabel('epoch')
plt.ylabel('test_acc')
plt.title('')
plt.plot(test_acc)
plt.annotate('(%.4f)'%test_acc[test_acc_max], xy = (test_acc_max, test_acc[test_acc_max]))
plt.show()



# Test Loss 
plt.xlabel('epoch')
plt.ylabel('test_loss')
plt.title('')
plt.plot(test_loss)
plt.annotate('(%.4f)'%test_loss[test_loss_min], xy = (test_loss_min, test_loss[test_loss_min]))
plt.show()

breakpoint()
