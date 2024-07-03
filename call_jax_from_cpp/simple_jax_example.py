# Useful references:
# 1. https://jax.readthedocs.io/en/latest/export/export.html.

import jax
from jax import export

def f(x):
    return x * x  / 100.0

def main():
    jit_f = jax.jit(f)

    dummy = 3. # The value is discarded.

    exported = export.export(jit_f)(dummy)
    serialized = exported.serialize()

    # TODO(joao): this does not appear to be useful; the C/C++ APIs seem to required a serialized ExecutableAndOptionsProto.
    with open("serialized_executables/example1.bytes", "wb") as bf:
        bf.write(serialized)

    hlo = jit_f.lower(dummy).compile().as_text()

    with open("hlo/example1.txt", "w") as hlo_f:
        hlo_f.write(hlo)

    # TODO(joao): this is required by the C API, but jax.xla_computation is deprecated.
    # Track question: https://github.com/google/jax/discussions/22266
    serialized_proto = jax.xla_computation(f)(dummy).as_serialized_hlo_module_proto()

    with open("hlo/example1.binpb", "wb") as bf:
        bf.write(serialized_proto)

if __name__ == "__main__":
    main()
