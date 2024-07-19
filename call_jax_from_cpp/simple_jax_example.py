# Useful references:
# 1. https://jax.readthedocs.io/en/latest/aot.html.
# 2. https://jax.readthedocs.io/en/latest/export/export.html
# 3. https://openxla.org/stablehlo/tutorials/jax-export

import jax

from jax import export


def f(x):
    return x * x / 100.0


def main():
    jit_f = jax.jit(f)

    dummy = 3.0  # The value is discarded.

    lowered = jit_f.lower(dummy)
    hlo = lowered.compile().as_text()

    with open("hlo/example1.txt", "w") as hlo_f:
        hlo_f.write(hlo)

    stable_hlo = export.export(jit_f)(dummy).mlir_module()

    with open("stable_hlo/example1.txt", "w") as hlo_f:
        hlo_f.write(stable_hlo)

    serialized_proto = lowered.compiler_ir("hlo").as_serialized_hlo_module_proto()

    with open("hlo/example1.binpb", "wb") as bf:
        bf.write(serialized_proto)


if __name__ == "__main__":
    main()
