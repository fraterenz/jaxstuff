# BlackJAX
BlackJAX is an MCMC sampling library based on JAX. BlackJAX provides well-tested and ready to use sampling algorithms.
For an example, see notebook `cox-process.ipynb`.

Remember that the log density has to be a pure JAX function (as JAX applies `grad` to it), so something like this won't work:

```python
def loglikelihood(my_key, t_indiv):
    gp = GaussianProcess(kernel, t_indiv, diag=1e-5)
    # array of shape T
    eta = jnp.exp(gp.sample(my_key))  # not a pure JAX fn
    return (gp.log_probability(t_indiv) + jnp.sum(jax.vmap(jax.scipy.stats.poisson.logpmf)(t_indiv, eta)))
```
we need to move the sampling outside of the computation by creating a new parameter, something like this:

```python
rng_key, subkey_init, subkey_indiv = jax.random.split(rng_key, 3)
keys_indiv = jax.random.split(subkey_indiv, burden.shape[0])
keys_init = jax.random.split(subkey_init, 3)

# remember at the end l = exp(u) and m = log(eta)
params = namedtuple("model_params", ["u", "sigma", "eta"])

def init_param(keys, t_indiv):
    key1, key2, key3 = keys[0], keys[1], keys[2] # hardcoded
    sigma = jax.random.uniform(key2)
    u = jax.random.uniform(key1)

    kernel = sigma** 2 * kernels.Matern32(jnp.exp(u))
    gp = GaussianProcess(kernel, t_indiv, diag=1e-5)
    # array of shape T
    m = gp.sample(key3) # TODO deal with this with Cholesky

    return params(
        u=u,  # TODO
        sigma=sigma,  #  TODO
        eta=jnp.exp(m),  #  TODO
    )
params_init = init_param(keys_init, age[idx == 0])
params_init

# at the indiv level
@jax.jit
def logdensity_fn(my_key, t_indiv, y_indiv, params):
    # hyperprior
    log_hyperprior = jax.scipy.stats.norm.logpdf(params.u)

    # (latent) prior
    kernel = params.sigma** 2 * kernels.Matern32(jnp.exp(params.u))
    gp = GaussianProcess(kernel, t_indiv, diag=1e-5)
    # array of shape T_indiv
    log_prior = gp.log_probability(jnp.log(params.eta)) # TODO fix eta

    # likelihood
    log_likelihood = jnp.sum(jax.scipy.stats.poisson.logpmf(y_indiv, params.eta))

    return log_hyperprior + log_prior + log_likelihood
```

## Sampling multiple chains
The best way to achieve this is to move the parallel computation from the instruction level (SIMD) to the program level, in JAX replacing `jax.vmap` with `jax.shard_map`.
This complicates the code and the computation as we move from automatic parallelisation with `vmap` to manually sharding data into devices with `jax.shard_map`.
However, this is required for algorithms that [require concurrency such as NUTS](https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html).
For example, every step of the MCMC performs a different number of deterministic integration steps.
Thus chains running in parallel spend different time performing the integration step, hence concurrency is needed otherwise the slowest chain will slow down all the other in SIMD settings.

Some code from `cox-process.ipynb`

```python
def run_one_chain(key, state, inference_algorithm, num_samples):
    # sharding with shard_map gives (1, D) per shard but
    # run_inference_algo requires (D, ) so squeezing is required
    key = jnp.squeeze(key, axis=0)
    state = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), state)
    _, (samples, info) = blackjax.util.run_inference_algorithm(
        key,
        inference_algorithm,
        num_samples,
        initial_state=state,
    )
    return samples, info


mesh = jax.make_mesh((MCMC_CHAINS,), ('chain',))
sharding = NamedSharding(mesh, P('chain'))

rng_key, sample_key = jax.random.split(rng_key)
# need to shard: 1. keys and 2. initial conditions
# 1. KEYS
sample_keys = jax.device_put(jax.random.split(sample_key, MCMC_CHAINS), sharding)
for shard in sample_keys.addressable_shards:
    print(shard.device, shard.index)
# 2. INITIAL CONDITIONS
# convoluted things to just repeat arrays MCMC_CHAINS times for sharding later
initial_params_repeated = jax.tree_util.tree_map(
    lambda e: jnp.broadcast_to(e, (MCMC_CHAINS, e.shape[0]) if not jax.numpy.isscalar(e) else (MCMC_CHAINS)),
    initial_params
)
initial_states = jax.vmap(
    nuts_adapted.init,
    in_axes=(0)
)(initial_params_repeated)
initial_states = jax.tree_map(
    lambda x: jnp.repeat(x[None], MCMC_CHAINS, axis=0),
    nuts_adapted.init(initial_params)
)
initial_states_sharded = jax.device_put(initial_states, sharding)
for shard in initial_states_sharded.position.u.addressable_shards:
    print(shard.device, shard.index)

run_one_chain_bound = partial(
    run_one_chain,
    inference_algorithm=nuts_adapted,
    num_samples=1_000 + BURNIN,
)

sharded_states, history = jax.jit(jax.shard_map(
    run_one_chain_bound,
    mesh=mesh,
    in_specs=(P('chain'), P('chain')),  # key, state
    out_specs=P('chain'),
    check_vma=False,
))(sample_keys, initial_states_sharded)

m_samples = sharded_states.position.m.reshape(MCMC_CHAINS, 1_000 + BURNIN, sharded_states.position.m.shape[-1])[:, BURNIN:]
```
