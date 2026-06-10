// Save
auto tensor = outputs->at(i);
tensorflow::TensorProto tensor_proto;
tensor.AsProtoTensorContent(&tensor_proto);

std::string file_name = debug_tensor_names[i];
for (const auto& c : {'/', ':'}) {
  auto p = file_name.find(c);
  if (p != std::string::npos) {
    file_name[p] = '_';
  }
}

std::ofstream ofs(file_name, std::ios::binary);
tensor_proto.SerializeToOstream(&ofs);
ofs.close();

# Load
tensorflow::TensorProto tensor_proto;
std::string file_name = "input_tensor_pb";
std::ifstream ifs(file_name, std::ios::binary);
tensor_proto.ParseFromIstream(&ifs);
ofs.close();
tensorflow::Tensor tensor;
tensor.FromProto(tensor_proto);
