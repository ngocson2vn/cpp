const auto& man_tensor = man_util_pred.pred();
const void* raw_data = man_tensor.content().data();
auto dtype = static_cast<tensorflow::DataType>(man_tensor.dtype());
const size_t elem_size = DataTypeSize(dtype);
const size_t num_bytes = man_tensor.content().size();
const size_t num_elements = num_bytes / elem_size;

tensorflow::Tensor tensor(dtype, tensorflow::TensorShape{num_elements});        
std::memcpy(tensor.data(), raw_data, num_bytes);