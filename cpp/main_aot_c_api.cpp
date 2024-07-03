// Useful references:
// 1. https://github.com/google/jax/discussions/22184
// 2. https://github.com/LeelaChessZero/lc0/blob/master/src/neural/xla/pjrt.cc

#include <fstream>
#include <iostream>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_cpu.h"

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
    // Read the serialized executable bytearray.
    const std::string inputFile = "serialized_executable_protos/example1.binpb";
    std::ifstream infile(inputFile, std::ios_base::binary);
    std::vector<char> bytes( (std::istreambuf_iterator<char>(infile)),
                              std::istreambuf_iterator<char>() );

    std::cout << "JOAO CKPT 1" << std::endl;

    // Get the API object.
    const PJRT_Api* api = GetPjrtApi();

    std::cout << "JOAO CKPT 2" << std::endl;

    // Create the client.
    auto client_create_args = MakeStruct<PJRT_Client_Create_Args>();
    CheckError(api->PJRT_Client_Create(&client_create_args), api);
    PJRT_Client* client = client_create_args.client;

    std::cout << "JOAO CKPT 3" << std::endl;

    // Get the loaded executable.
    auto deserialize_args = MakeStruct<PJRT_Executable_DeserializeAndLoad_Args>();
    deserialize_args.client = client;
    deserialize_args.serialized_executable = bytes.data();
    deserialize_args.serialized_executable_size = bytes.size();
    CheckError(api->PJRT_Executable_DeserializeAndLoad(&deserialize_args), api);
    PJRT_LoadedExecutable* loaded_executable = deserialize_args.loaded_executable;

    std::cout << "JOAO CKPT 4" << std::endl;

    // Get the executable.
    auto load_executable_args = MakeStruct<PJRT_LoadedExecutable_GetExecutable_Args>();
    load_executable_args.loaded_executable = loaded_executable;
    CheckError(api->PJRT_LoadedExecutable_GetExecutable(&load_executable_args), api);
    PJRT_Executable* executable = load_executable_args.executable;

    std::cout << "JOAO CKPT 5" << std::endl;

    // Get the number of outputs.
    auto num_outputs_args = MakeStruct<PJRT_Executable_NumOutputs_Args>();
    num_outputs_args.executable = executable;
    CheckError(api->PJRT_Executable_NumOutputs(&num_outputs_args), api);
    const size_t num_outputs = num_outputs_args.num_outputs;
    std::vector<PJRT_Buffer*> outputs(num_outputs);

    std::cout << "JOAO CKPT 6" << std::endl;

    // TODO(joao): continue below.
    // 1. Execute.

    // struct PJRT_LoadedExecutable_Execute_Args {
    //   size_t struct_size;
    //   PJRT_Extension_Base* extension_start;
    //   PJRT_LoadedExecutable* executable;
    //   // Only needs to stay alive for the duration of the Execute call.
    //   PJRT_ExecuteOptions* options;
    //   // Execution input of size [`num_devices`, `num_args`].
    //   PJRT_Buffer* const* const* argument_lists;
    //   size_t num_devices;
    //   size_t num_args;
    //   // Execution output of size [`num_devices`, num_outputs`], where `num_outputs`
    //   // is the number of outputs returned by this executable per device. Both the
    //   // outer (`PJRT_Buffer***`) and inner lists (`PJRT_Buffer**`) must be
    //   // allocated and deallocated by the caller. PJRT_Buffer_Destroy must be called
    //   // on the output PJRT_Buffer*.
    //   PJRT_Buffer** const* output_lists;  // in/out
    //   // If `device_complete_events` isn't nullptr, `device_complete_events` needs
    //   // to be the same length as `output_lists` (i.e. of length `num_devices`), and
    //   // each `PJRT_Event` will become ready once the corresponding device execution
    //   // is complete. If Execute returns an error, then `device_complete_events`
    //   // will not be populated. The caller is responsible for calling
    //   // PJRT_Event_Destroy on the returned PJRT_Event*s.
    //   PJRT_Event** device_complete_events;  // in/out
    //   // The device to execute on. If nullptr, will execute on the device(s)
    //   // specified at compile time. If set, must be an addressable device, and
    //   // `num_devices` should be 1 with `argument_lists` only containing arguments
    //   // for `execute_device`. Can be set with a multi-device executable to launch
    //   // just on this device. In this case, it's the responsibility of the caller to
    //   // make sure the executable is launched on all participating devices specified
    //   // at compile time. Setting this field may not be supported on all platforms
    //   // or executables.
    //   PJRT_Device* execute_device;
    // };
    // PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Execute_Args, execute_device);
    //
    // // Executes on devices addressable by the client.
    // typedef PJRT_Error* PJRT_LoadedExecutable_Execute(
    //     PJRT_LoadedExecutable_Execute_Args* args);

    // auto args = MakeStruct<PJRT_LoadedExecutable_Execute_Args>();
    // args.executable = loaded_executable;
    // args.options = &options;
    // args.num_devices = 1;
    // std::vector<PJRT_Buffer*> buffers(inputs.size());
    // for (size_t i = 0; i < inputs.size(); ++i) buffers[i] = inputs[i]->buffer_;
    // PJRT_Buffer* const* buffers_ptr = buffers.data();
    // args.num_args = inputs.size();
    // args.argument_lists = &buffers_ptr;

    // PJRT_Buffer** outputs_ptr = outputs.data();
    // PJRT_Event* event_ptr;
    // args.output_lists = &outputs_ptr;
    // args.device_complete_events = &event_ptr;
    // CheckError(api->PJRT_LoadedExecutable_Execute(&args));

    return 0;
}
