// Useful references:
// 1. https://github.com/LeelaChessZero/lc0/blob/51f93b7c49720ee100d24aac54193a88ba98219a/src/neural/xla/pjrt.cc#L212

#include <fstream>
#include <iostream>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_cpu.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/hlo.pb.h"

template <typename T>
T MakeStruct() {
  T t;
  memset(&t, 0, sizeof(t));
  t.struct_size = sizeof(t);
  return t;
}

PJRT_Error_Code GetErrorCode(const PJRT_Api* api, PJRT_Error* error) {
  auto args = MakeStruct<PJRT_Error_GetCode_Args>();
  args.error = error;
  api->PJRT_Error_GetCode(&args);
  return args.code;
}

void CheckError(PJRT_Error* error, const PJRT_Api* api) {
  if (!error) return;
  const PJRT_Error_Code code = GetErrorCode(api, error);
  std::cout << "RECEIVED ERROR CODE " << code << std::endl;
  throw;
}

int main(int argc, char** argv) {
    std::cout << "CKPT 1" << std::endl;

    // Get the API object.
    const PJRT_Api* api = GetPjrtApi();

    std::cout << "CKPT 2" << std::endl;

    // Create the client.
    auto client_create_args = MakeStruct<PJRT_Client_Create_Args>();
    CheckError(api->PJRT_Client_Create(&client_create_args), api);
    PJRT_Client* client = client_create_args.client;

    std::cout << "CKPT 3" << std::endl;

    // Read the HLO file.
    // constexpr std::string_view kFilename = "hlo/example1.txt";
    constexpr std::string_view kFilename = "hlo/example1.binpb";
    constexpr std::string_view kFormat = "hlo";

    std::ifstream in(kFilename);
    const std::string hlo((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    std::cout << "CKPT 4" << std::endl;

    // Construct the HLO program.
    auto program = MakeStruct<PJRT_Program>();
    program.code = const_cast<char*>(hlo.data());
    program.code_size = hlo.size();
    program.format = kFormat.data();
    program.format_size = kFormat.size();

    std::cout << "CKPT 5" << std::endl;

    // Compile the HLO program.
    xla::CompileOptions compile_options;
    xla::CompileOptionsProto compile_options_proto = compile_options.ToProto().value();

    const size_t num_bytes_serialized_compile_options_proto = compile_options_proto.ByteSizeLong();
    std::vector<char> byte_array_serialized_compile_options_proto(num_bytes_serialized_compile_options_proto);
    compile_options_proto.SerializeToArray(byte_array_serialized_compile_options_proto.data(),
                                           num_bytes_serialized_compile_options_proto);

    std::cout << "CKPT 6" << std::endl;  // TODO(joao): fails after this.

    // TODO(joao): we should be able to get this to work, as the C++ API example works.

    auto compile_args = MakeStruct<PJRT_Client_Compile_Args>();
    compile_args.client = client;
    compile_args.program = &program;
    compile_args.compile_options = byte_array_serialized_compile_options_proto.data();
    compile_args.compile_options_size = num_bytes_serialized_compile_options_proto;
    CheckError(api->PJRT_Client_Compile(&compile_args), api);
    PJRT_LoadedExecutable* loaded_executable = compile_args.executable;

    std::cout << "CKPT 7" << std::endl;

    // Get the executable.
    auto load_executable_args = MakeStruct<PJRT_LoadedExecutable_GetExecutable_Args>();
    load_executable_args.loaded_executable = loaded_executable;
    CheckError(api->PJRT_LoadedExecutable_GetExecutable(&load_executable_args), api);
    PJRT_Executable* executable = load_executable_args.executable;

    std::cout << "CKPT 8" << std::endl;

    auto serialization_args = MakeStruct<PJRT_Executable_Serialize_Args>();
    serialization_args.executable = executable;
    CheckError(api->PJRT_Executable_Serialize(&serialization_args), api);

    std::cout << "CKPT 9" << std::endl;

    // Run `find -L . -name example1.bytes` to the file (Bazel things).
    std::ofstream ofs("example1.bytes", std::ios_base::out | std::ios_base::binary);
    ofs.write(serialization_args.serialized_bytes, serialization_args.serialized_bytes_size);  

    return 0;
}
