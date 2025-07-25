import os
import jax
import jax.numpy as jnp
from jax import random, jit, pmap
from jax import tree_util
from functools import partial

# Print SLURM environment for debugging
print("ENV:", {k: v for k, v in os.environ.items() if k.startswith("SLURM")})

# -------------------
# Multi-node setup
# -------------------
if "SLURM_PROCID" in os.environ:
    coordinator = f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=int(os.environ["WORLD_SIZE"]),
        process_id=int(os.environ["SLURM_PROCID"]),
    )
print(
    f"Process {jax.process_index()} ready on "
    f"{jax.local_device_count()} local devices."
)

# -------------------
# Dummy model: simple linear regression
# -------------------
def init_params(key, in_dim, out_dim):
    w_key, b_key = random.split(key)
    return {
        "w": random.normal(w_key, (in_dim, out_dim)),
        "b": random.normal(b_key, (out_dim,))
    }

def forward(params, x):
    return jnp.dot(x, params["w"]) + params["b"]

def loss_fn(params, x, y):
    pred = forward(params, x)
    return jnp.mean((pred - y) ** 2)

# -------------------
# Simple SGD (no optax)
# -------------------
learning_rate = 1e-3

def sgd_update(params, grads, lr):
    return tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

@jit
def train_step(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    grads = jax.lax.pmean(grads, axis_name='batch')  # average across devices
    params = sgd_update(params, grads, learning_rate)
    return params, loss

# -------------------
# Setup
# -------------------
key = random.PRNGKey(0)
in_dim, out_dim = 128, 10
params = init_params(key, in_dim, out_dim)

# Fake data
batch_size = 1024
x_data = random.normal(key, (batch_size, in_dim))
y_data = random.normal(key, (batch_size, out_dim))

# -------------------
# PMAP (data parallel)
# -------------------
params = jax.device_put_replicated(params, jax.local_devices())
x_data = jax.device_put_replicated(x_data, jax.local_devices())
y_data = jax.device_put_replicated(y_data, jax.local_devices())

@partial(pmap, axis_name='batch')
def train_step_pmap(params, x, y):
    return train_step(params, x, y)

# -------------------
# Training loop
# -------------------
for epoch in range(1000):
    params, loss = train_step_pmap(params, x_data, y_data)
    if jax.process_index() == 0:
        print(f"Epoch {epoch}: loss {jnp.mean(loss)}")
