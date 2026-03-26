# BlackJAX
BlackJAX is an MCMC sampling library based on JAX. BlackJAX provides well-tested and ready to use sampling algorithms.

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

