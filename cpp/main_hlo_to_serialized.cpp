// Useful references:
// 1. https://github.com/openxla/xla/issues/7038

#include <fstream>
#include <iostream>

#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/tools/hlo_module_loader.h"

int main(int argc, char** argv) {
    // Load HloModule from file.
    std::string hlo_filename = "hlo/example1.txt";
    std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
        [](xla::HloModuleConfig* config) { config->set_seed(42); };
    std::unique_ptr<xla::HloModule> test_module =
        xla::LoadModuleFromFile(
            hlo_filename,
            "txt",
            xla::hlo_module_loader_details::Config(),
            config_modifier_hook
        ).value();
    const xla::HloModuleProto test_module_proto = test_module->ToProto();

    // Get a CPU client.
    std::unique_ptr<xla::PjRtClient> client = xla::GetTfrtCpuClient(true).value();

    // Compile XlaComputation to PjRtExecutable.
    xla::XlaComputation xla_computation(test_module_proto);
    xla::CompileOptions compile_options;
    std::unique_ptr<xla::PjRtLoadedExecutable> executable =
        client->Compile(xla_computation, compile_options).value();
    const std::string serialized = executable->SerializeExecutable().value();
    xla::ExecutableAndOptionsProto proto;
    proto.set_serialized_executable(serialized);
    *proto.mutable_compile_options() = compile_options.ToProto().value();

    // Run `find -L . -name example1.binpb` to the file (Bazel things).
    std::ofstream ofs("example1.binpb", std::ios_base::out | std::ios_base::binary);
    const bool succeeded = proto.SerializeToOstream(&ofs);
    std::cout << "Succeeded? " << succeeded << std::endl;

    return 0;
}
