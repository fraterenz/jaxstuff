# JAX
A summary:

- funcional language: immutable arrays? TODO 101 and Quickstart and crash course
- pytrees
- autodiff (see below)
- `jax.lax`
- sharding
- asynchronous dispatch
- accelerators: JAX → jaxpr → StableHLO MLIR → XLA optimizations/codegen → CPU/GPU/TPU executable (see below)
- https://docs.jax.dev/en/latest/notebooks/vmapped_log_probs.html

## Pytrees
TODO.
Ref: https://docs.jax.dev/en/latest/custom_pytrees.html#pytrees-custom-pytree-nodes

## Autodiff
TODO, ref:
1. https://docs.jax.dev/en/latest/gradient-checkpointing.html
2. https://docs.jax.dev/en/latest/automatic-differentiation.html
3. https://docs.jax.dev/en/latest/advanced-autodiff.html#advanced-autodiff
4. https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html

## `jax.lax`
This library contains primitives operations that underpins libraries (e.g. `jax.numpy`).
Transformation rules, such as JVP and batching rules, are typically defined as transformations
on jax.lax primitives.

Many of the primitives are thin wrappers around equivalent XLA operations, described by the XLA
operation semantics documentation.
In a few cases JAX diverges from XLA, usually to ensure that the set of operations is closed
under the operation of JVP and transpose rules.

XLA (Accelerated Linear Algebra): an open-source compiler for machine learning.

### `jax.lax.scan`
TODO

### MapReduce
The
The reduction must be monoidal (the operation must be associative, and the initial value must be
an identity wrt that operation), or the result is undefined. Since the reduction is monoidal, it
can be parallised via tree reduction, and JAX will do this automatically.

## Sharding
https://docs.jax.dev/en/latest/sharded-computation.html

## Async dispatch
[TODO](https://docs.jax.dev/en/latest/async_dispatch.html).

## Compilation, IR and other interesting things
Mostly from [ahead-of-time doc](https://docs.jax.dev/en/latest/aot.html).

### Basics of compilation in JAX
1. Stage out a specialized version of the original Python callable F to an internal representation (see below).
2. Lower this specialized, staged-out computation to the XLA compiler’s input language, StableHLO (as an MLIR module).
3. Compile the lowered HLO program to produce an optimized executable for the target device (CPU, GPU, or TPU).
4. Execute the compiled executable with the arrays x and y as arguments.

MLIR is a common IR that also supports hardware specific operations and it's a LLVM product.
As an analogy,
- Rust → IR → LLVM IR → LLVM optimizations/codegen → machine code
and similarly
- JAX → jaxpr → StableHLO MLIR → XLA optimizations/codegen → CPU/GPU/TPU executable

#### IR and jaxpr
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

**why aot?**
1. Predictable latency: precompile during startup/deploy, then execution is “ready to go” (no surprise first-call compile).
2. Control/inspection: grab the jaxpr, StableHLO text, cost analysis, etc., without executing.
3. Deployment workflows: if you need to move computations across processes or archive them, you typically use jax.export rather than raw lowered StableHLO
But, AOT-compiled functions cannot be transformed by JAX’s just-in-time transformations such as `jax.jit`,
`jax.grad()`, and `jax.vmap()`.

Compiled functions are specialized to a particular set of argument with types, such as arrays with a specific
shape and element type in our running example. From JAX’s internal point of view, transformations such as
`jax.vmap()` alter the type signature of functions in a way that invalidates the compiled-for type signature.

