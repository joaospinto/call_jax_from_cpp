# Introduction

This repository shows examples of running JAX models from C++ code.

One option (which involves JIT compilation) is to use an HLO file to run the JAX mode.
This can be tried by running `bazel run //cpp:hlo_example`.

Another approach (relying on AOT compilation) is to serialize a pre-compiled executable
beforehand. Once the executable exists, this can be achieved by running `bazel run //cpp:aot_example`.

# Setting up the Python environment

Run the following commands:
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -e .`

# Generating the HLO and serialized files

`python3 call_jax_from_cpp/simple_jax_example.py`
