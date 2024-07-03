cc_binary(
    name = "hlo_example",
    srcs = ["main_hlo.cpp"],
    deps = [
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tools:xla_compile_lib",
    ],
    data = [
        "//hlo:hlo",
    ],
)

cc_binary(
    name = "aot_example",
    srcs = ["main_aot.cpp"],
    deps = [
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@xla//xla/tools:xla_compile_lib",
    ],
    data = [
        "//serialized_executables:serialized_executables",
    ],
)

cc_binary(
    name = "serialize_hlo",
    srcs = ["main_hlo_to_serialized.cpp"],
    deps = [
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@xla//xla/tools:xla_compile_lib",
    ],
    data = [
        "//hlo:hlo",
    ],
)
