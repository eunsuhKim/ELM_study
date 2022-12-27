#%%
import numpy as np
import matplotlib.pyplot as plt

tl = 0.0
tr = 1.0
xl = 0.0
xr = 1.0

nt = 101
nx = 32

kappa = 3.0

xs = np.linspace(xl, xr, nx)
ts = np.linspace(tl, tr, nt)
Xs, Ts = np.meshgrid(xs, ts)
Us = np.sin(np.pi*Xs)*np.exp(-np.pi**2/kappa* Ts)

plt.figure(figsize=(10,8),facecolor='white')
plt.contourf(Xs, Ts, Us, 100, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# %%
t_snap = [0.0,0.20,0.40,0.60,0.80,1.00] # even number
plt.figure(figsize=(10,12),tight_layout=True)
for i in range(0,len(t_snap)):
    t_val = t_snap[i]
    plt.subplot(3,2, i+1)
    plt.title(f"t={t_val}",fontsize=20)
    plt.plot(xs, Us[int((nt-1)*(t_val/(tr-tl))),:],lw=3,color='r')
    plt.xlabel('x',fontsize=15)
    plt.ylabel('u',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.ylim([0,1.05])
plt.show()

# %%
from scipy.io import savemat
saving_dict = {}
saving_dict['x'] = xs
saving_dict['t'] = ts
saving_dict['usol'] = Us
savemat(f"heat_diffusion_kappa_{kappa}.mat",saving_dict)
# %%
