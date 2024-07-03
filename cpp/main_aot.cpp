// Useful references:
// 1. https://github.com/google/jax/discussions/22184
// 2. https://jax.readthedocs.io/en/latest/aot.html
// 3. https://jax.readthedocs.io/en/latest/export/index.html

#include <fstream>
#include <iostream>
#include <iterator>

#include "xla/literal_util.h"
#include "xla/pjrt/cpu/cpu_client.h"

int main(int argc, char** argv) {
    // Read the serialized executable bytearray.
    const std::string inputFile = "serialized_executables_proto/example1.pbbin";
    std::ifstream infile(inputFile, std::ios_base::binary);
    std::vector<char> bytes( (std::istreambuf_iterator<char>(infile)),
                              std::istreambuf_iterator<char>() );
    std::string_view serialized_view(bytes.data(), bytes.size());

    // Get a CPU client.
    std::unique_ptr<xla::PjRtClient> client = xla::GetTfrtCpuClient(true).value();

    std::unique_ptr<xla::PjRtLoadedExecutable> executable = client->DeserializeExecutable(serialized_view, std::nullopt).value();

    // Input.
    xla::Literal literal_x;
    std::unique_ptr<xla::PjRtBuffer> param_x;
    // Output.
    std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results;
    std::shared_ptr<xla::Literal> result_literal;
    xla::ExecuteOptions execute_options;

    for(int i = 0; i < 1000; i++) {
        literal_x = xla::LiteralUtil::CreateR2<float>({{0.1f * (float)i}});
        param_x = client->BufferFromHostLiteral(
            literal_x, client->addressable_devices()[0]
        ).value();

        results = executable->Execute({{param_x.get()}}, execute_options).value();
        result_literal = results[0][0]->ToLiteralSync().value();
        std::cout << "Result " << i << " = " << *result_literal << "\n";
    }

    return 0;
}
