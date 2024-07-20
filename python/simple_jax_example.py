# Useful references:
# 1. https://jax.readthedocs.io/en/latest/aot.html.
# 2. https://jax.readthedocs.io/en/latest/export/export.html
# 3. https://jax.readthedocs.io/en/latest/jax.stages.html#jax.stages.Lowered.compiler_ir
# 4. https://jax.readthedocs.io/en/latest/_autosummary/jax.export.Exported.html#jax.export.Exported.mlir_module_serialized
# 5. https://jax.readthedocs.io/en/latest/export/export.html#export-calling-convention-version
# 6. https://openxla.org/stablehlo/tutorials/jax-export
# 7. https://openxla.org/stablehlo/vhlo
# 8. https://openxla.org/stablehlo/compatibility

import jax

from jax import export


def f(x):
    return x * x / 100.0


def main():
    jit_f = jax.jit(f)

    dummy = 3.0  # The value is discarded.

    lowered = jit_f.lower(dummy)
    exported = export.export(jit_f)(dummy)

    # These are intended as debug-only serializations. Not portable!
    hlo = lowered.compiler_ir("hlo").as_hlo_text()
    serialized_hlo_proto = lowered.compiler_ir("hlo").as_serialized_hlo_module_proto()
    stable_hlo = str(lowered.compiler_ir("stablehlo"))

    # This is portable.
    mlir_bc = exported.mlir_module_serialized

    with open("hlo/example1.txt", "w") as hlo_f:
        hlo_f.write(hlo)

    with open("hlo/example1.binpb", "wb") as hlo_bf:
        hlo_bf.write(serialized_hlo_proto)

    with open("stable_hlo/example1.txt", "w") as stable_hlo_f:
        stable_hlo_f.write(stable_hlo)

    with open("mlir/example1.bc", "wb") as bc_f:
        bc_f.write(mlir_bc)


if __name__ == "__main__":
    main()
