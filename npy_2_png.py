import numpy as np
import matplotlib.pyplot as plt

npy = np.load('./1016_test_1400/Deep_SVDD_Loss.npy')
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
# plt.yscale('log')
plt.plot(npy)
plt.show()