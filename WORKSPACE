workspace(name = "call_compiled_jax_from_cpp")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Load XLA.

http_archive(
    name = "xla",
    urls = ["https://github.com/openxla/xla/archive/refs/heads/main.zip"],
    strip_prefix = "xla-main",
)

load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.11": "//build:requirements_lock_3_11.txt",
        "3.12": "//build:requirements_lock_3_12.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()
