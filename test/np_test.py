import numpy as np
# np.set_printoptions(threshold=np.nan)

batch_size = 28
seq_length = 30
latent_dim = 5

sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))

print(sample.shape)
print(sample)

sample[:, :, 0] = np.linspace(0, 1.0/seq_length, num=seq_length)

print(sample.shape)
print(sample)

# print(sample[:, :, 0].shape)
# print(sample[:, :, 0])

