# Just JAX
<img src="https://media1.tenor.com/m/e61tmVht2w4AAAAd/just-jack-sean-hayes.gif" width="200" height="150" alt="a man in a yellow shirt is sitting at a table with a drink in his hand and making a funny face ." style="max-width: 833px;" fetchpriority="high">

## Key points to remember
JAX is a language for expressing and composing transformations of numerical programs.

1. JAX main goal is to transform functions such that we can leverage different devices, async computation, vectorisation, sharding and XLA.
1. JAX → jaxpr → StableHLO MLIR → XLA optimizations/codegen → CPU/GPU/TPU executable
2. JAX → jaxpr requires not only a function but also tracing its input
3. the `jit` decorator registers the fn as just-in-time but doesn't compile it yet: `jit` functions are compiled just-in-time once per input (tracing), and recompiled every time the function is called with input of a new type and/or shape (`ShapedArray((3,), jnp.float32)`)
4. `jit` control flow is evaluated at compile time (jit as usual but with the input)
4. autodiff
5. local random state makes code thread safe
1. JAX array objects are represented as DeviceArray instances (having a device attribute) and are agnostic to the place where the array lives
(CPU, GPU, or TPU)
6. DeviceArray instances are actually futures due to the default asynchronous execution in JAX, Python call might return before the computation
actually ends (`block_until_ready`)
7. JAX arrays are tracers (symbolic placeholders), no concrete values
7. [out-of-bound indices errors](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing) not raised `0_0`
and thus it's undefined behavior.

## Pure functions
Pure function (JAX): a Python function that does not read global external state or write to external state.
A function can be functionally pure even if it actually uses stateful objects internally:
```python

def pure_uses_internal_state(x):
  state = dict(even=0, odd=0)
  for i in range(10):
    state['even' if i % 2 == 0 else 'odd'] += x
  return state['even'] + state['odd']


print(jit(pure_uses_internal_state)(5.))
```

The idea is that jaxpr does not capture the side-effect present in a function, but captures the function as
executed on the parameters given to it.
For example, if we have a Python conditional, the jaxpr will only know about the branch we take.
JAX re-runs the Python function when the type or shape of the argument changes, but not the value:
```python
g = 0.
def impure_uses_globals(x):
  return x + g


# JAX captures the value of the global during the first run
print ("First call: ", jit(impure_uses_globals)(4.))
# First call:  4.0

g = 10.  # Update the global
# Subsequent runs may silently use the cached value of the globals
print ("Second call: ", jit(impure_uses_globals)(5.))
# Second call:  5.0, since jax hasn't recompiled the fn as 4. and 5. have same type

# JAX re-runs the Python function when the type or shape of the argument changes
# This will end up reading the latest value of the global
print ("Third call, different type: ", jit(impure_uses_globals)(jnp.array([4.])))
Third call, different type:  [14.]
```

Arrays are immutable and can be mutated by returning a new copy of the originial array with the update value
using `at`, but the original array will be the same.
This is memory-heavy.
However, inside jit-compiled code, if the input value `x` of `x.at[idx].set(y)` is not reused, the compiler
will optimize the array update to occur in-place.


**debug.**
If you want debug printing, use `jax.debug.print()`.
To express general side-effects at the cost of performance, see `jax.experimental.io_callback()`.
To check for tracer leaks at the cost of performance, use with `jax.check_tracer_leaks())`.

### Statefull to stateless programs
In JAX the transformations (not only `jit`) needs to run on pure function.
Transform statefull programs
```python
class StatefulClass
  state: State
  def stateful_method(*args, **kwargs) -> Output:
```
into stateless programs
```python
class StatelessClass
  def stateless_method(state: State, *args, **kwargs) -> (Output, State):
```
Is it possible to write fast loops with state, using `scan` and `associative_scan`.
For example
```python
from jax import lax
import operator, jax.numpy as jnp

x = jnp.array([1., 2., 3.])
prefix = lax.associative_scan(operator.add, x)
# prefix = [1., 3., 6.]
```
See also this [video](https://www.youtube.com/watch?v=NlQ1N3W3Wms).

## `vmap`
TODO.
The reduction must be monoidal (the operation must be associative, and the initial value must be an identity wrt that operation), or the result is undefined. Since the reduction is monoidal, it can be parallised via tree reduction, and JAX will do this automatically.


## Pseudo random nb gen
JAX implements an explicit PRNG where entropy production and consumption are handled by explicitly passing and
iterating a PRNG state. JAX uses a modern Threefry counter-based PRNG that’s splittable. That is, its design allows
us to fork the PRNG state into new PRNGs for use with parallel stochastic generation.
The state is represented as a pair of two unsigned-int32s that is called a key:
```python
key = random.PRNGKey(0)
print(key)
# Array([0, 0], dtype=uint32)
```

The code is thread safe, since the local random state eliminates possible race conditions involving global state.
`jax.random.split` is a deterministic function that converts one key into several independent (in the pseudorandomness sense)
keys.

## `jax.lax`
This library contains primitives operations that underpins libraries (e.g. `jax.numpy`).
A primitive is a fundamental unit of computation used in JAX programs.
Most functions in jax.lax represent individual primitives.
When representing a computation in a jaxpr, each operation in the jaxpr is a primitive.
Transformation rules, such as JVP and batching rules, are typically defined as transformations
on jax.lax primitives.

Many of the primitives are thin wrappers around equivalent XLA operations, described by the XLA
operation semantics documentation.
In a few cases JAX diverges from XLA, usually to ensure that the set of operations is closed
under the operation of JVP and transpose rules.


### `jax.lax.scan`
The doc says the Haskell-like type signature is
```haskell
scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
```
To understand this, break it down into smaller parts.
`scan` takes
1. a function `f` with signature `f :: (c, a) -> (c, b)` which is equivalent to `f :: (c -> a -> (c, b))` according to ChatGTP
2. a carry `c` and an array of things `[a]`
and returns the carry `c` with another array of things `[b]`.

It’s like a “fold” (reduce) that also records an output at every step.
Note that the loop-carried value carry must hold a fixed shape and dtype across all iterations.
`scan` compiles `f`, so while it can be combined with `jit`, it’s usually unnecessary.
`scan` is designed for iterating with a static number of iterations (compare with `fori_loop()` or `while_loop()`).

**IMPORTANT!** When your per-step output is a PyTree, JAX will stack each leaf # across time, which looks like a transpose from
“sequence of pytrees” to “pytree of sequences”! Example:
```python
# scan :: (f, (c, [a])) -> (c, [b])
# f :: (c, a) -> (c, b)
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):  # f :: (c, a) -> (c, b)
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    # (f, (c, [a])) -> (c, [b])
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    # note that b is a PyTree (per-step output) so JAX will stack each leaf
    # across time, which looks like a transpose from “sequence of pytrees”
    # to “pytree of sequences”

    return states, infos
```
here the output of `scan` should be a list of tuples `[(state0, info0), (state1, info1), ...]`, but since `b` is a PyTree,
we get instead what we want, i.e. `([state0, state1, ...], [info0, info1, ...])`.
The `b` PyTree from `f` dictates the PyTree output of the scanned function.


### MapReduce
The
The reduction must be monoidal (the operation must be associative, and the initial value must be
an identity wrt that operation), or the result is undefined. Since the reduction is monoidal, it
can be parallised via tree reduction, and JAX will do this automatically.


## Tracing
Tracing: turn Python into a graph/program (jaxpr) by following objects to determine the sequence of operations performed
by a Python function.
This is necessary for later converting Python code to jaxpr, not just for `jit`.
jaxpr is JAX’s traced IR, and it's obtained by calling transformations (`jit`, `grad`, `vmap` etc) on functions.
JAX does tracing based on some inputs.
All JAX operations being expressed in terms of XLA, as `jax.lax` operations are Python
wrappers for operations in XLA.
XLA (Accelerated Linear Algebra): an open-source compiler for machine learning.

A function can be traced with an abstract input `ShapedArray((3,), jnp.float32)`.
This ensures that we cache the compiled function for each concrete value of a float32 Array with one axis and shape 3.
It will recompiled if we use the same function if we use that function with an array of different type and/or shape.

There are [two types of tracers](https://docs.jax.dev/en/latest/tracing.html#different-kinds-of-jax-values):
1. `ShapedArray` with information about the shape and dtype of an array,
2. `ConcreteArray` include the regular array data, need for example for resolving conditionals.

Almost all JAX's transformations work with abstract tracers, and exception is `grad` which uses concrete tracers.
So if we want to `jit` a function `def divide(x, y): return x / y if y >= 1. else 0` we need to tell jax to use
concrete values for `y`, using `static_argnums=1`.

### Static vs traced values
Static means known at compile time (shape and type, not values).
JAX re-runs the Python function when the type or shape of the argument changes, but not the value:
```python
g = 0.
def impure_uses_globals(x):
  return x + g


# JAX captures the value of the global during the first run
print ("First call: ", jit(impure_uses_globals)(4.))
# First call:  4.0

g = 10.  # Update the global
# Subsequent runs may silently use the cached value of the globals

print ("Second call: ", jit(impure_uses_globals)(5.))
# Second call:  5.0, since jax hasn't recompiled the fn as 4. and 5. have same type


# JAX re-runs the Python function when the type or shape of the argument changes

# This will end up reading the latest value of the global
print ("Third call, different type: ", jit(impure_uses_globals)(jnp.array([4.])))
Third call, different type:  [14.]
```

How does it work:
```python
selu_jit = jax.jit(selu)

# Pre-compile the function before timing...
selu_jit(x).block_until_ready()


%timeit selu_jit(x).block_until_ready()
```
1. We defined `selu_jit` as the compiled version of `selu`, but haven't compiled yet.
2. We called `selu_jit` once on x (tracing, see below).
3. We timed the execution speed of the compiled version. (Note the use of `block_until_ready()`, which is required due to JAX’s Asynchronous dispatch).

### Static vs traced operations
From [this](https://docs.jax.dev/en/latest/tracing.html#static-vs-traced-operations).

Just as values can be either static or traced, operations can be static or traced.
Static operations are evaluated at compile-time in Python; traced operations are compiled & evaluated at run-time in XLA.
Consider this example here, which fails:
```python
import jax.numpy as jnp
from jax import jit

@jit
def f(x):
  return x.reshape(jnp.array(x.shape).prod())  # x.reshape((np.prod(x.shape),))

x = jnp.ones((2, 3))
f(x)
```
the problem is that shapes must be static in `jnp.array`.
Here `x.shape` is static, and there is no problem when executing `jnp.array(x.shape=`).
But the `jnp.prod` is a JAX operation within a Tracer-context (`jit`, `vmap`, `grad` etc...), which results in a Tracer (a placeholder array, not
concrete values), which messes up with `x.reshape`.
We can see that this operation (function) transforms static data into Tracer.
For this reason, a standard convention in JAX programs is to import numpy as np and import jax.numpy as jnp so that both interfaces are available
for finer control over whether operations are performed in a static manner (with numpy, once at compile-time) or a traced manner (with jax.numpy,
optimized at run-time).

## JIT
Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once.
JIT compiles the program into a fast executable and cache it.
Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time.

The `jit`-jaxpr is then compiled using XLA into very efficient code optimized for your GPU or TPU.
Finally, the compiled code is executed to satisfy the call.
Subsequent calls to `selu_jit` will use the compiled code directly, skipping the python implementation entirely.
If we didn’t include the warm-up call separately, everything would still work, but then the compilation time would
be included in the benchmark.
It would still be faster, because we run many loops in the benchmark, but it wouldn’t be a fair comparison.

**why `jit`?**
Let's consider an example: `grad` vs `jit` and `grad`.
When using `grad` on a function with many primitives (small atomic operations like sin, cos, sum, etc.), the code executes as Python with JAX dispatching
all the primitives individually.
Those primitives are still executed by the backend (typically via XLA-compiled executables), but each op / small cluster is dispatched separately → lots
of Python, that is device overhead and less fusion.
On the other hand, by `jit`ting the `grad(f)`, we do something different.
JAX still lowers to XLA but in only a single pass: JAX traces the entire gradient function and lowers + compiles it as one XLA computation.
Since the fun is compiled into an executable, `jit`ted functions bypass all the python overheads/calls and result into a single fused
executable (read low-level function, where all the small primitives got fused into a single operation kind of).

A great example from GTP5.2:
```python
import time
import jax
import jax.numpy as jnp

# Optional: shows when *whole-function* compilation happens
jax.config.update("jax_log_compiles", True)

def f(x):
    y = x

    for _ in range(10):
        y = jnp.sin(y) + 0.1 * jnp.cos(y * y) - jnp.tanh(y)
    return jnp.sum(y)

g  = jax.grad(f)      # traces f to build grad, but runs as Python dispatching ops
gj = jax.jit(g)       # traces + compiles the *entire gradient* into one cached executable

x = jnp.ones((1_000_000,), dtype=jnp.float32)

print("jaxpr(grad):")
print(jax.make_jaxpr(g)(x))     # lots of primitives (sin/cos/mul/add/...) due to loop unrolling

print("\njaxpr(jit(grad)):")
print(jax.make_jaxpr(gj)(x))    # ONE call primitive (pjit/xla_call)

# Warmup (first gj call includes compile)
g(x).block_until_ready()
gj(x).block_until_ready()

t0 = time.perf_counter(); g(x).block_until_ready();  t1 = time.perf_counter()
t2 = time.perf_counter(); gj(x).block_until_ready(); t3 = time.perf_counter()

print("\nsecond-call runtime (grad):     ", t1 - t0)
print("second-call runtime (jit(grad)):", t3 - t2)

# Optional: inspect the compiled program; look for fusion, fewer launched computations, etc.
print("\nCompiled IR (stablehlo):")
print(gj.lower(x).compiler_ir(dialect="stablehlo"))
```

### Control flow
Control flow (if/else) is tricky with `jit` (but not with `grad`).
We need to know the result of the boolean evaluation at compile time, but this will change at runtime.
Indeed, the abstract input (`ShapedArray((3,), jnp.float32)`) won't help us here, because the branch of the control flow taken by the
program will depend on the **values** of the concrete array, not shape nor type.
Traced values within JIT can only affect control flow via their static attributes: such as shape or dtype, and not via their values.

One solution is to trace on concrete values, use `jit(f, static_argnames='x')`, see [here](https://docs.jax.dev/en/latest/control-flow.html).
This works well, however will recompile for each new `x` no matter the type or the shape.
To actually get `jit` behaviour, we can use `jax.lax` primitives for [control flow](https://docs.jax.dev/en/latest/control-flow.html#structured-control-flow-primitives).
Have a look at `jax.lax.cond` and all the different variants such as `jax.lax.{select,switch}` and `jax.numpy.{where,piecewise,select}`.
And check out the [summary](https://docs.jax.dev/en/latest/control-flow.html#summary) as well.
We also have access to logical operators such as `logical_{and,or,not}`.

**loops:** with `jit` the loops can be handled in the following ways based on the trip count (the number of iterations to perform):
1. unrolling with Python loops with small, *fixed* (means known at `jit` compile-time) trip counts,
2. if large trip counts, prefer `fori_loop`/`scan` to avoid huge graphs,
3. for loops with data-dependent trip counts / early stopping, use `while_loop`.
If you want to collect outputs over time or want reverse-mode AD behavior consider `lax.scan`.

**`lax.scan`:** see above.

## Pytrees
Pytree is a recursive datatype that represents a container for data.
A pytree can either be:
1. a container node that JAX knows how to traverse, like list, tuple, dict, any custom type you register, another pytree,
2. a leaf (a value JAX treats as atomic), like a JAX/NumPy array, a scalar, an object, etc.
Any object whose type is not in the pytree container registry will be treated as a leaf node in the tree.

Interestingly, `jax.tree.leaves` will completely flatten the tree:
```python
example_trees = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]
# [1, 'a', <object object at 0x78a992bf4050>, 1, 2, 3, 1, 2, 3, 4, 5, 2, 2, 3, Array([1, 2, 3], dtype=int32)]
print(jax.tree.leaves(example_trees)) # compare to the for loop below

# Print how many leaves the pytrees have.
for pytree in example_trees:
  # This `jax.tree.leaves()` method extracts the flattened leaves from the pytrees.
  leaves = jax.tree.leaves(pytree)
  print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")
```
However, `map` [preserves](https://docs.jax.dev/en/latest/pytrees.html#common-function-jax-tree-map) the tree structure.

You can register your structure at runtime as a pytree and then leverage [JAX API](https://docs.jax.dev/en/latest/jax.tree_util.html),
which is very useful for JAX's tranform functions.
These functions work on the `children` data, not on the metdata `aux_data`: `children, aux_data = flatten(x)`.
Rule of thumb: if changing it should trigger recompilation / be static, put it in aux_data; if it’s numeric state you want transformed,
put it in children.
To [register your datatype](https://docs.jax.dev/en/latest/custom_pytrees.html#custom-pytree-nodes) use
```python
register_pytree_node(
    RegisteredSpecial,
    special_flatten,    # Instruct JAX what are the children nodes.
    special_unflatten   # Instruct JAX how to pack back into a `RegisteredSpecial`.
)
```
Unlike `NamedTuple` subclasses, classes decorated with `@dataclass` are not automatically pytrees, see
[here](https://docs.jax.dev/en/latest/custom_pytrees.html#custom-pytree-nodes).
Some datatypes (e.g. `str`) cannot be `jit` (not known at compile time), so we need to put them as metadata in `aux_data` outside
of the leaves.
Pytrees are tree-like, rather than DAG-like or graph-like, in that we handle them assuming referential transparency and that they
can’t contain reference cycles.

You can specify the axis of a tree to apply some computation.
For example, `jax.vmap(f, in_axes=(None, {"k1": None, "k2": 1}))(mytree)` means apply f to the tree only on leaf `k2` on the axis
number 1.

### key path
In a pytree each leaf has a key path.
A key path for a leaf is a list of keys, where the length of the list is equal to the depth of the leaf in the pytree.
For example, `tree = [1, {'k1': 2, 'k2': (3, 4)}, ATuple('foo')]` the key path of `1` (first leaf of `tree`), is `[0]`,
hence we can do:
```python
import collections

ATuple = collections.namedtuple("ATuple", ('name'))

tree = [1, {'k1': 2, 'k2': (3, 4)}, ATuple('foo')]
flattened, _ = jax.tree_util.tree_flatten_with_path(tree)

for key_path, value in flattened:
  print(f'Value of tree{jax.tree_util.keystr(key_path)}: {value}')
```
which prints
```python
Value of tree[0]: 1
Value of tree[1]['k1']: 2
Value of tree[1]['k2'][0]: 3
Value of tree[1]['k2'][1]: 4
Value of tree[2].name: foo
```
To express key paths, JAX provides a few default key types for the built-in pytree node types, namely:
1. `SequenceKey(idx: int)`: For lists and tuples.
2. `DictKey(key: Hashable)`: For dictionaries.
3. `GetAttrKey(name: str)`: For namedtuples and preferably custom pytree nodes (more in the next section)
For example,
```python
for key_path, _ in flattened:
  print(f'Key path of tree{jax.tree_util.keystr(key_path)}: {repr(key_path)}')
```
which prints
```python
Key path of tree[0]: (SequenceKey(idx=0),)
Key path of tree[1]['k1']: (SequenceKey(idx=1), DictKey(key='k1'))
Key path of tree[1]['k2'][0]: (SequenceKey(idx=1), DictKey(key='k2'), SequenceKey(idx=0))
Key path of tree[1]['k2'][1]: (SequenceKey(idx=1), DictKey(key='k2'), SequenceKey(idx=1))
Key path of tree[2].name: (SequenceKey(idx=2), GetAttrKey(name='name'))
```

## Autodiff (AD)
Autodiff converts a program into another program, JAX's `grad`.
`grad` is another useful transformation that takes a function and compute the gradient of that function by translating Python code
into jaxpr.
Without `jit` the `grad` XLA compilation is not triggered and instructions are translated using jaxpr only (like `vmap`).

`grad` functions are translated into primitive jaxpr expressions, augmenting the computation by tracing also the derivatives
of the intermediate values (see below for more information): every primitive autodiff applies the chain rule locally.
It works both on forward mode (better suited when input's dimensionality is smaller than outputs' dimensionality) and backward
mode (for the contrary), see below.

### Notes on AD
See "Automatic Differentiation in Machine Learning: a Survey".
AD works by keeping track (trace) intermediate variables and their derivatives, which enables to compute the derivative of
computer programs (with control flows, loops, recursion etc) as opposed to symbolic differentiation.
From the paper *"any numeric code will eventually result in a numeric evaluation trace with particular values of the input,
intermediate, and output variables, which are the only things one needs to know for computing derivatives using chain rule
composition, regardless of the specific control flow path that was taken during execution"*.
AD provides numerical values of derivatives (as opposed to derivative expressions) by using symbolic rules of differentiation
(but keeping track of derivative values as opposed to the resulting expressions), giving it a two-sided nature that is partly
symbolic and partly numerical.

To compute partial derivatives with respect to a vector v (that is directional derivatives), we can apply the forward or
reverse mode, or a combination of the two:
- partial first derivative $\text{D}_v (x) = \text{J}(x) v $, use the forward mode
- gradient of the first directional derivative $\nabla^2 f v = \text{H}_f v $, apply the reverse mode to take the gradient produced by the forward mode
- partial second derivative $\text{D}^2_v (x) = v^T \text{H} v $, use the forward mode and then ??

Note that `grad` is implemented in reverse mode.
So when you want to compute the direction derivative $\nabla f v$ don't do `jnp.dot(grad(f), v)` but instead do `jvp(f, (x,), (v,))`.
Similarly, if you want to compute $\text{H}v$ (Hessian vector product, HVP) use forward over reverse `jvp(grad(f), X, V)`, and not
`jnp.tensordot(hessian(f)(X), V, 2)`.

**forward mode:** intermediate variables are augmented by their derivative with respect to the input dimensions.
We then compute the derivative of the intermediate variables with respect to the inputs.
Thus works well when the dimensionality of the inputs is low.
Very efficient to compute the product between the Jacobian and a vector because TODO.
The doc says primals and tangents, where primals are the points where the Jacobian is evaluated in the Taylor's expansion.
We start in the domain and go into the codomain.

**backward mode:** still need a forward pass (as in the forward mode) to compute the intermediate variables.
We then compute the derivative of the intermediate variables with respect to the outputs (like back propagation).
Thus works well when the dimensionality of the output is low (like loss in NN).
We start in the codomain and go into the domain.


**ref:**
1. https://docs.jax.dev/en/latest/gradient-checkpointing.html
2. https://docs.jax.dev/en/latest/automatic-differentiation.html
3. https://docs.jax.dev/en/latest/advanced-autodiff.html#advanced-autodiff
4. https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html
5. this great [lecture](https://videolectures.net/videos/deeplearning2017_johnson_automatic_differentiation)

## Sharding
https://docs.jax.dev/en/latest/sharded-computation.html
https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html

## Async dispatch
JAX uses asynchronous dispatch to hide Python overheads.
The main idea: run Python code in the CPU while the GPU/TPU does the computation.
If using CPU only, then the host is the Python thread and the device is the CPU JAX backend runtime, which can run compiled
work in native code.

From [this](https://docs.jax.dev/en/latest/async_dispatch.html).

When an operation such as `jnp.dot(x, x)` is executed, JAX does not wait for the operation to complete before returning
control to the Python program.
Instead, JAX returns a jax.Array value, which is a future, i.e. a value that will be produced in the future on an accelerator
device but isn’t necessarily available immediately.
We can inspect the shape or type of a array without waiting for the computation that produced it to complete, and we can even pass it
to another JAX computation, as we do with the addition operation here.
Only if we actually inspect the value of the array from the host, for example by printing it or by converting it into a plain old
`numpy.ndarray` will JAX force the Python code to wait for the computation to complete.

Asynchronous dispatch is useful since it allows Python code to “run ahead” of an accelerator device, keeping Python code out of the critical
path.
Provided the Python code enqueues work on the device faster than it can be executed, and provided that the Python code does not actually need
to inspect the output of a computation on the host, then a Python program can enqueue arbitrary amounts of work and avoid having the
accelerator wait.


## Compilation, IR and other interesting things
Python work: control flow, object creation, tracing, dispatch.
Native work: the heavy numeric kernels running outside the interpreter.

Mostly from [ahead-of-time doc](https://docs.jax.dev/en/latest/aot.html).

1. Stage out a specialized version of the original Python callable F to an internal representation (see below).
2. Lower this specialized, staged-out computation to the XLA compiler’s input language, StableHLO (as an MLIR module).
3. Compile the lowered HLO program to produce an optimized executable for the target device (CPU, GPU, or TPU).
4. Execute the compiled executable with the arrays x and y as arguments.

MLIR is a common IR that also supports hardware specific operations and it's a LLVM product.
As an analogy,
- Rust → IR → LLVM IR → LLVM optimizations/codegen → machine code
and similarly
- JAX → jaxpr → StableHLO MLIR → XLA optimizations/codegen → CPU/GPU/TPU executable

All JAX operations being expressed in terms of XLA, as `jax.lax` operations are Python
wrappers for operations in XLA.

Arrays in JAX are represented as DeviceArray instances and are agnostic to the place where the array lives (CPU, GPU, or TPU).

### IR and jaxpr
JAX code gets translated into an intermediate representation (IR) called jaxpr, which is a language per se.
Many `jax.lax` functions are basically Python wrappers around primitives in jaxpr.
This IR is not exactly a DAG (GTP says), since JAX IR can include control-flow primitives (`lax.cond`, `lax.while_loop`,
`scan`), which embed nested jaxprs; with loops, the overall program isn’t a single DAG.

Stage out: it takes the Python function F, run it in “tracing” mode with abstract inputs (shapes/dtypes), and
records the sequence of JAX primitives it performs into a compiler-friendly graph/IR (e.g., jaxpr → StableHLO),
producing a specialized version for those input shapes/dtypes.
So you’re not executing F to get numbers; you’re executing it to build a program.
The specialization reflects a restriction of F to input types inferred from properties of the arguments x and y
(usually their shape and element type). JAX carries out this specialization by a process that we call tracing.
During tracing, JAX stages the specialization of F to a jaxpr, which is a function in the Jaxpr intermediate language.


### Accelerating code
If you’re using accelerators, using NumPy arrays directly will result in multiple transfers from CPU to GPU/TPU memory.
You can save that transfer bandwidth, either by creating directly a DeviceArray or by using jax.device_put on the NumPy array.
With DeviceArrays, computation is done on device so no additional data transfer is required, e.g.
`jnp.dot(long_vector, long_vector)` will only transfer a single scalar (result of the computation) back from device to host.

Remember aot compilation vs just-in-time compilation: compilation occurs before the program
runs (before execution) vs compilation occurs at runtime.
JAX can do both :0 jit and aot compiled code.
It can even send code to accelerators (GPU, TPUs) in both situations, that is aot and jit compiled code.

To do that, for jit code, use the `jax.jit`.

For aot, `jax.jit(F).lower(...).compile()`, where `lower` takes objects:
```python
def f(x, y): return 2 * x + y
x, y = 3, 4
traced = jax.jit(f).trace(x, y)
lowered_with_x_y = jax.jit(f).trace(x, y).lower().compile()(x, y)
# or a container for the shape, dtype, and other static attributes of an array
i32_scalar = jax.ShapeDtypeStruct((), jnp.dtype('int32'))
lowered_with_x_y = jax.jit(f).trace(i32_scalar, i32_scalar).lower().compile()(x, y)
```

AOT gives you a function specialized to (shapes/dtypes + static-arg values).
We have control/debug of these functionality thanks to the JAX’s AOT API (impressive):
```python
import jax

def f(x, y): return 2 * x + y

x, y = 3, 4

traced = jax.jit(f).trace(x, y)
# Print the specialized, staged-out representation (as Jaxpr IR)
print(traced.jaxpr)

lowered = traced.lower()
# Print lowered HLO
print(lowered.as_text())

compiled = lowered.compile()

# Query for cost analysis, print FLOP estimate
compiled.cost_analysis()['flops']

# Execute the compiled function!
compiled(x, y)
```

#### why aot?
1. Predictable latency: precompile during startup/deploy, then execution is “ready to go” (no surprise first-call compile).
2. Control/inspection: grab the jaxpr, StableHLO text, cost analysis, etc., without executing.
3. Deployment workflows: if you need to move computations across processes or archive them, you typically use jax.export rather than raw lowered StableHLO
But, AOT-compiled functions cannot be transformed by JAX’s just-in-time transformations such as `jax.jit`,
`jax.grad()`, and `jax.vmap()`.

Compiled functions are specialized to a particular set of argument with types, such as arrays with a specific
shape and element type in our running example. From JAX’s internal point of view, transformations such as
`jax.vmap()` alter the type signature of functions in a way that invalidates the compiled-for type signature.

