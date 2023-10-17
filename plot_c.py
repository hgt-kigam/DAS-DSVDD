import torch
import numpy as np

state_dict = torch.load('./1016_test_1400/AE_best_save.pth')
c = state_dict['center']
c = np.array(c)
print(c)
print(c.shape)
