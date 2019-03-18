import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.genfromtxt("new_gal")
data1 = pd.read_csv('new_gal', sep=" ")

# data zSource logM e_logM

mu_obs = data1["logM"].values # logM
error = data1["e_logM"].values # e_logM
redshift = data1["zSN"].values # zSource

plt.scatter(redshift, mu_obs, marker='.', color='navy', alpha=0.5)
plt.show()