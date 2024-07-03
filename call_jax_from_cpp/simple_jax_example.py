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

    with open("serialized_executables/example1.bytes", "wb") as bf:
        bf.write(serialized)

    hlo = jit_f.lower(dummy).compile().as_text()

    with open("hlo/example1.txt", "w") as hlo_f:
        hlo_f.write(hlo)

if __name__ == "__main__":
    main()
