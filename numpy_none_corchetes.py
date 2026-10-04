import numpy as np

x = np.zeros((128, 2))
print(x[None].shape)          # (1, 128, 2)
print(x[:, None].shape)       # (128, 1, 2)
print(x[:, :, None].shape)    # (128, 2, 1)

# Broadcasting, igual que en PyTorch
p = np.random.randn(5, 2)
q = np.random.randn(7, 2)
dif = p[:, None, :] - q[None, :, :]       # (5, 7, 2)
dist = np.linalg.norm(dif, axis=-1)       # (5, 7)
print(dist.shape)