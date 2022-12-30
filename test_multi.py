#%%
import jax.numpy as jnp
import numpy as onp
# %%
A = onp.random.randn(1,3,2,4)
x = onp.random.randn(2,4)
# %%
print((A@x.T).shape)
# (1,3,2,4) @ (4,2) 
# = (1,3,2,2)
# %%
print((A.reshape(1*3,2*4)@x.reshape(2*4,1)).shape)
# %%
A_ = A.reshape(1*3,2*4)
x_ = x.reshape(2*4,1)


# %%
(A_.T@A_@x_).shape
#%%
b = onp.random.randn(1,3)
b_ = b.reshape(1*3,1)
#%%
(A_.T@b_).shape
# %%
X_ = jnp.array(x_)

# %%
A_ @ X_
# %%
