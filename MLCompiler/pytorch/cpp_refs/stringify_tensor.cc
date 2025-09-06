    try {
      fprintf(stderr, "Copy tensor_name: %s\n", tensor_name.c_str());
      // tiling tensor, cpu:shape[0] == gpu:shape[0]
      // normal tensor, gpu:shape[0] == max_batch_size
      gpu_tensor.slice(0, 0, cpu_tensor.size(0)).copy_(cpu_tensor, true);
    } catch (const std::exception& ex) {
      std::string msg = std::string("an exception occurred when copying tensor ")
                            .append(tensor_name)
                            .append(" src shape ")
                            .append(c10::str(cpu_tensor.sizes()))
                            .append(" dst shape ")
                            .append(c10::str(gpu_tensor.sizes()));
                            // .append(": ")
                            // .append(ex.what());
      throw std::runtime_error(msg);
    }