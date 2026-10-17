module attributes {gpu.container_module} {
  func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
    cf.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = arith.muli %dim, %c8 : index
    %3 = arith.muli %2, %c1024 : index
    %4 = arith.divsi %3, %c2048 : index
    %5 = arith.ceildivsi %3, %c2048 : index
    gpu.launch_func  @predict_online_5603970_0_kernel::@predict_online_5603970_0_kernel blocks in (%5, %c1, %c1) threads in (%c512, %c1, %c1) args(%4 : index, %arg1 : memref<?x8x1024xf32>, %0 : memref<?x8x1024xf32>, %3 : index)
    return %0 : memref<?x8x1024xf32>
  ^bb2:  // pred: ^bb0
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %6 = tf_framework.null_memref : memref<?x8x1024xf32>
    return %6 : memref<?x8x1024xf32>
  }
  gpu.module @predict_online_5603970_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    llvm.func @predict_online_5603970_0_kernel(%arg0: i32, %arg1: !llvm.ptr<f32>, %arg2: !llvm.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: !llvm.ptr<f32>, %arg11: !llvm.ptr<f32>, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(-1 : index) : i32
      %1 = llvm.mlir.constant(8192 : index) : i32
      %2 = llvm.mlir.constant(2048 : index) : i32
      %3 = llvm.mlir.constant(512 : index) : i32
      %4 = llvm.mlir.constant(1024 : index) : i32
      %5 = llvm.mlir.constant(1536 : index) : i32
      %6 = llvm.mlir.constant(8 : index) : i32
      %7 = llvm.mlir.constant(5.000000e-01 : f32) : f32
      %8 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %9 = nvvm.read.ptx.sreg.ctaid.x : i32
      %10 = nvvm.read.ptx.sreg.tid.x : i32
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %11 = llvm.mul %9, %2  : i32
      %12 = llvm.icmp "slt" %9, %arg0 : i32
      llvm.cond_br %12, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %13 = llvm.add %10, %3  : i32
      %14 = llvm.add %10, %4  : i32
      %15 = llvm.add %10, %5  : i32
      %16 = llvm.add %10, %11  : i32
      %17 = llvm.add %13, %11  : i32
      %18 = llvm.add %14, %11  : i32
      %19 = llvm.add %15, %11  : i32
      %20 = llvm.srem %16, %4  : i32
      %21 = llvm.srem %17, %4  : i32
      %22 = llvm.srem %18, %4  : i32
      %23 = llvm.srem %19, %4  : i32
      %24 = llvm.sdiv %16, %4  : i32
      %25 = llvm.sdiv %17, %4  : i32
      %26 = llvm.sdiv %18, %4  : i32
      %27 = llvm.sdiv %19, %4  : i32
      %28 = llvm.srem %24, %6  : i32
      %29 = llvm.srem %25, %6  : i32
      %30 = llvm.srem %26, %6  : i32
      %31 = llvm.srem %27, %6  : i32
      %32 = llvm.sdiv %24, %6  : i32
      %33 = llvm.sdiv %25, %6  : i32
      %34 = llvm.sdiv %26, %6  : i32
      %35 = llvm.sdiv %27, %6  : i32
      %36 = llvm.mul %32, %1  : i32
      %37 = llvm.mul %28, %4  : i32
      %38 = llvm.add %36, %37  : i32
      %39 = llvm.add %38, %20  : i32
      %40 = llvm.getelementptr %arg2[%39] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %41 = llvm.load %40 : !llvm.ptr<f32>
      %42 = llvm.fmul %41, %7  : f32
      %43 = llvm.mul %33, %1  : i32
      %44 = llvm.mul %29, %4  : i32
      %45 = llvm.add %43, %44  : i32
      %46 = llvm.add %45, %21  : i32
      %47 = llvm.getelementptr %arg2[%46] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %48 = llvm.load %47 : !llvm.ptr<f32>
      %49 = llvm.fmul %48, %7  : f32
      %50 = llvm.mul %34, %1  : i32
      %51 = llvm.mul %30, %4  : i32
      %52 = llvm.add %50, %51  : i32
      %53 = llvm.add %52, %22  : i32
      %54 = llvm.getelementptr %arg2[%53] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %55 = llvm.load %54 : !llvm.ptr<f32>
      %56 = llvm.fmul %55, %7  : f32
      %57 = llvm.mul %35, %1  : i32
      %58 = llvm.mul %31, %4  : i32
      %59 = llvm.add %57, %58  : i32
      %60 = llvm.add %59, %23  : i32
      %61 = llvm.getelementptr %arg2[%60] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %62 = llvm.load %61 : !llvm.ptr<f32>
      %63 = llvm.fmul %62, %7  : f32
      %64 = "llvm.intr.maximum"(%42, %8) : (f32, f32) -> f32
      %65 = llvm.getelementptr %arg11[%39] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %64, %65 : !llvm.ptr<f32>
      %66 = "llvm.intr.maximum"(%49, %8) : (f32, f32) -> f32
      %67 = llvm.getelementptr %arg11[%46] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %66, %67 : !llvm.ptr<f32>
      %68 = "llvm.intr.maximum"(%56, %8) : (f32, f32) -> f32
      %69 = llvm.getelementptr %arg11[%53] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %68, %69 : !llvm.ptr<f32>
      %70 = "llvm.intr.maximum"(%63, %8) : (f32, f32) -> f32
      %71 = llvm.getelementptr %arg11[%60] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %70, %71 : !llvm.ptr<f32>
      llvm.br ^bb12
    ^bb3:  // pred: ^bb1
      %72 = llvm.mul %11, %0  : i32
      %73 = llvm.add %72, %arg19  : i32
      %74 = llvm.icmp "slt" %73, %3 : i32
      %75 = llvm.select %74, %73, %3 : i1, i32
      %76 = llvm.icmp "slt" %10, %75 : i32
      llvm.cond_br %76, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %77 = llvm.add %10, %11  : i32
      %78 = llvm.srem %77, %4  : i32
      %79 = llvm.sdiv %77, %4  : i32
      %80 = llvm.srem %79, %6  : i32
      %81 = llvm.sdiv %79, %6  : i32
      %82 = llvm.mul %81, %1  : i32
      %83 = llvm.mul %80, %4  : i32
      %84 = llvm.add %82, %83  : i32
      %85 = llvm.add %84, %78  : i32
      %86 = llvm.getelementptr %arg2[%85] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %87 = llvm.load %86 : !llvm.ptr<f32>
      %88 = llvm.fmul %87, %7  : f32
      %89 = "llvm.intr.maximum"(%88, %8) : (f32, f32) -> f32
      %90 = llvm.getelementptr %arg11[%85] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %89, %90 : !llvm.ptr<f32>
      llvm.br ^bb5
    ^bb5:  // 2 preds: ^bb3, ^bb4
      %91 = llvm.add %11, %3  : i32
      %92 = llvm.mul %91, %0  : i32
      %93 = llvm.add %92, %arg19  : i32
      %94 = llvm.icmp "slt" %93, %3 : i32
      %95 = llvm.select %94, %93, %3 : i1, i32
      %96 = llvm.add %10, %3  : i32
      %97 = llvm.icmp "slt" %10, %95 : i32
      llvm.cond_br %97, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %98 = llvm.add %96, %11  : i32
      %99 = llvm.srem %98, %4  : i32
      %100 = llvm.sdiv %98, %4  : i32
      %101 = llvm.srem %100, %6  : i32
      %102 = llvm.sdiv %100, %6  : i32
      %103 = llvm.mul %102, %1  : i32
      %104 = llvm.mul %101, %4  : i32
      %105 = llvm.add %103, %104  : i32
      %106 = llvm.add %105, %99  : i32
      %107 = llvm.getelementptr %arg2[%106] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %108 = llvm.load %107 : !llvm.ptr<f32>
      %109 = llvm.fmul %108, %7  : f32
      %110 = "llvm.intr.maximum"(%109, %8) : (f32, f32) -> f32
      %111 = llvm.getelementptr %arg11[%106] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %110, %111 : !llvm.ptr<f32>
      llvm.br ^bb7
    ^bb7:  // 2 preds: ^bb5, ^bb6
      %112 = llvm.add %11, %4  : i32
      %113 = llvm.mul %112, %0  : i32
      %114 = llvm.add %113, %arg19  : i32
      %115 = llvm.icmp "slt" %114, %3 : i32
      %116 = llvm.select %115, %114, %3 : i1, i32
      %117 = llvm.add %10, %4  : i32
      %118 = llvm.icmp "slt" %10, %116 : i32
      llvm.cond_br %118, ^bb8, ^bb9
    ^bb8:  // pred: ^bb7
      %119 = llvm.add %117, %11  : i32
      %120 = llvm.srem %119, %4  : i32
      %121 = llvm.sdiv %119, %4  : i32
      %122 = llvm.srem %121, %6  : i32
      %123 = llvm.sdiv %121, %6  : i32
      %124 = llvm.mul %123, %1  : i32
      %125 = llvm.mul %122, %4  : i32
      %126 = llvm.add %124, %125  : i32
      %127 = llvm.add %126, %120  : i32
      %128 = llvm.getelementptr %arg2[%127] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %129 = llvm.load %128 : !llvm.ptr<f32>
      %130 = llvm.fmul %129, %7  : f32
      %131 = "llvm.intr.maximum"(%130, %8) : (f32, f32) -> f32
      %132 = llvm.getelementptr %arg11[%127] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %131, %132 : !llvm.ptr<f32>
      llvm.br ^bb9
    ^bb9:  // 2 preds: ^bb7, ^bb8
      %133 = llvm.add %11, %5  : i32
      %134 = llvm.mul %133, %0  : i32
      %135 = llvm.add %134, %arg19  : i32
      %136 = llvm.icmp "slt" %135, %3 : i32
      %137 = llvm.select %136, %135, %3 : i1, i32
      %138 = llvm.add %10, %5  : i32
      %139 = llvm.icmp "slt" %10, %137 : i32
      llvm.cond_br %139, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %140 = llvm.add %138, %11  : i32
      %141 = llvm.srem %140, %4  : i32
      %142 = llvm.sdiv %140, %4  : i32
      %143 = llvm.srem %142, %6  : i32
      %144 = llvm.sdiv %142, %6  : i32
      %145 = llvm.mul %144, %1  : i32
      %146 = llvm.mul %143, %4  : i32
      %147 = llvm.add %145, %146  : i32
      %148 = llvm.add %147, %141  : i32
      %149 = llvm.getelementptr %arg2[%148] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %150 = llvm.load %149 : !llvm.ptr<f32>
      %151 = llvm.fmul %150, %7  : f32
      %152 = "llvm.intr.maximum"(%151, %8) : (f32, f32) -> f32
      %153 = llvm.getelementptr %arg11[%148] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %152, %153 : !llvm.ptr<f32>
      llvm.br ^bb11
    ^bb11:  // 2 preds: ^bb9, ^bb10
      llvm.br ^bb12
    ^bb12:  // 2 preds: ^bb2, ^bb11
      llvm.return
    }
  }
}

module attributes {gpu.container_module} {
  func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
    cf.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = arith.muli %dim, %c8 : index
    %3 = arith.muli %2, %c1024 : index
    %4 = arith.divsi %3, %c2048 : index
    %5 = arith.ceildivsi %3, %c2048 : index
    gpu.launch_func  @predict_online_5603970_0_kernel::@predict_online_5603970_0_kernel blocks in (%5, %c1, %c1) threads in (%c512, %c1, %c1) args(%4 : index, %arg1 : memref<?x8x1024xf32>, %0 : memref<?x8x1024xf32>, %3 : index) {uselessIndex = dense<[7, 14, 9, 16, 4, 18, 6, 13, 1, 8, 15, 3, 10, 17, 5, 12]> : tensor<16xi32>}
    return %0 : memref<?x8x1024xf32>
  ^bb2:  // pred: ^bb0
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %6 = tf_framework.null_memref : memref<?x8x1024xf32>
    return %6 : memref<?x8x1024xf32>
  }
  gpu.module @predict_online_5603970_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>, gpu.binary = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00z\00\00\00\00\00\00\00\00\00\00\00\C0\1E\00\00\00\00\00\00\00\1B\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0F\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.uft.entry\00.nv.info\00.text.predict_online_5603970_0_kernel\00.nv.info.predict_online_5603970_0_kernel\00.nv.shared.predict_online_5603970_0_kernel\00.nv.constant0.predict_online_5603970_0_kernel\00.debug_frame\00.debug_line\00.nv_debug_line_sass\00.rel.nv_debug_line_sass\00.nv_debug_ptx_txt\00.rel.debug_frame\00.rela.debug_frame\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.uft.entry\00.nv.info\00predict_online_5603970_0_kernel\00.text.predict_online_5603970_0_kernel\00.nv.info.predict_online_5603970_0_kernel\00.nv.shared.predict_online_5603970_0_kernel\00.nv.constant0.predict_online_5603970_0_kernel\00_param\00.debug_frame\00.debug_line\00.nv_debug_line_sass\00.rel.nv_debug_line_sass\00.nv_debug_ptx_txt\00.rel.debug_frame\00.rela.debug_frame\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00`\00\00\00\03\00\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DA\00\00\00\03\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\1C\01\00\00\03\00\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\01\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00T\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\89\01\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\12\10\0E\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\04\04\00\00\00\04\18\00\00\00\0C\81\80\80(\00\04H\01\00\00\00\00\00\00\00\00\00\16\00\00\00\02\00\10\00\00\00\01\01\FB\0E\0A\00\01\01\01\01\00\00\00\01\00\00\E2\00\00\00\02\00\10\00\00\00\01\01\FB\0E\0A\00\01\01\01\01\00\00\00\01\00\00\00\09\02\00\00\00\00\00\00\00\00\04\00\03\10\01\03\0B\02\10\01\03u\02\10\01\03\0C\02\10\01\F1\EE\F1\03\0A\02\10\01\EA\EC\F7\F0\03x\02\10\01\F1\F5\F0\F5\03t\02\10\01\F5\F2\ED\F5\F2\F7\03i\02\10\01\03\12\02\10\01\03w\02\10\01\ED\F6\EE\F4\EE\F2\F1\03z\02\10\01\F2\F1\F2\F0\F2\F4\ED\ED\F0\F3\F1\F4\ED\03\11\02\10\01\03\09\02\10\01\03i\02\10\01\03\18\02\10\01\03w\02\10\01\03r\02\10\01\03\0F\02\10\01\03\09\02\10\01\03b\02\10\01\EB\F4\F0\F4\F2\03\01\02 \01\03\02\02 \01\03\01\02 \01\ED\03\03\02 \01\F0\F0\03\05\020\01\F0\F0\F0\F0\03\05\02 \01\F1\F2\F0\ED\F2\F0\F0\F1\02\F0\01\00\01\01\00\00\00\00.version 7.1\00.target sm_86\00.address_size 64\00\00\00\00.visible .entry predict_online_5603970_0_kernel(\00.param .u32 predict_online_5603970_0_kernel_param_0,\00.param .u64 predict_online_5603970_0_kernel_param_1,\00.param .u64 predict_online_5603970_0_kernel_param_2,\00.param .u32 predict_online_5603970_0_kernel_param_3\00)\00{\00.reg .pred \09%p<6>;\00.reg .b32 \09%r<23>;\00.reg .f32 \09%f<25>;\00.reg .b64 \09%rd<29>;\00\00ld.param.u32 \09%r4, [predict_online_5603970_0_kernel_param_0];\00ld.param.u64 \09%rd7, [predict_online_5603970_0_kernel_param_2];\00cvta.to.global.u64 \09%rd1, %rd7;\00ld.param.u64 \09%rd8, [predict_online_5603970_0_kernel_param_1];\00cvta.to.global.u64 \09%rd2, %rd8;\00mov.u32 \09%r5, %ctaid.x;\00mov.u32 \09%r1, %tid.x;\00shl.b32 \09%r2, %r5, 11;\00setp.ge.s32 \09%p1, %r5, %r4;\00@%p1 bra \09$L__BB0_2;\00add.s32 \09%r20, %r1, %r2;\00or.b32  \09%r21, %r2, %r1;\00add.s32 \09%r22, %r20, 1536;\00mul.wide.s32 \09%rd17, %r21, 4;\00add.s64 \09%rd18, %rd2, %rd17;\00ld.global.f32 \09%f13, [%rd18];\00mul.f32 \09%f14, %f13, 0f3F000000;\00cvt.s64.s32 \09%rd19, %r2;\00cvt.u64.u32 \09%rd20, %r1;\00add.s64 \09%rd21, %rd20, %rd19;\00shl.b64 \09%rd22, %rd21, 2;\00add.s64 \09%rd23, %rd2, %rd22;\00ld.global.f32 \09%f15, [%rd23+2048];\00mul.f32 \09%f16, %f15, 0f3F000000;\00ld.global.f32 \09%f17, [%rd18+4096];\00mul.f32 \09%f18, %f17, 0f3F000000;\00mul.wide.s32 \09%rd24, %r22, 4;\00add.s64 \09%rd25, %rd2, %rd24;\00ld.global.f32 \09%f19, [%rd25];\00mul.f32 \09%f20, %f19, 0f3F000000;\00max.NaN.f32 \09%f21, %f14, 0f00000000;\00add.s64 \09%rd26, %rd1, %rd17;\00st.global.f32 \09[%rd26], %f21;\00max.NaN.f32 \09%f22, %f16, 0f00000000;\00add.s64 \09%rd27, %rd1, %rd22;\00st.global.f32 \09[%rd27+2048], %f22;\00max.NaN.f32 \09%f23, %f18, 0f00000000;\00st.global.f32 \09[%rd26+4096], %f23;\00max.NaN.f32 \09%f24, %f20, 0f00000000;\00add.s64 \09%rd28, %rd1, %rd24;\00st.global.f32 \09[%rd28], %f24;\00bra.uni \09$L__BB0_10;\00$L__BB0_2:\00ld.param.u32 \09%r3, [predict_online_5603970_0_kernel_param_3];\00sub.s32 \09%r6, %r3, %r2;\00min.s32 \09%r7, %r6, 512;\00setp.ge.s32 \09%p2, %r1, %r7;\00or.b32  \09%r8, %r2, %r1;\00mul.wide.s32 \09%rd9, %r8, 4;\00add.s64 \09%rd3, %rd1, %rd9;\00add.s64 \09%rd4, %rd2, %rd9;\00@%p2 bra \09$L__BB0_4;\00ld.global.f32 \09%f1, [%rd4];\00mul.f32 \09%f2, %f1, 0f3F000000;\00max.NaN.f32 \09%f3, %f2, 0f00000000;\00st.global.f32 \09[%rd3], %f3;\00$L__BB0_4:\00add.s32 \09%r10, %r6, -512;\00min.s32 \09%r11, %r10, 512;\00setp.ge.s32 \09%p3, %r1, %r11;\00@%p3 bra \09$L__BB0_6;\00cvt.s64.s32 \09%rd10, %r2;\00cvt.u64.u32 \09%rd11, %r1;\00add.s64 \09%rd12, %rd11, %rd10;\00shl.b64 \09%rd13, %rd12, 2;\00add.s64 \09%rd14, %rd1, %rd13;\00add.s64 \09%rd15, %rd2, %rd13;\00ld.global.f32 \09%f4, [%rd15+2048];\00mul.f32 \09%f5, %f4, 0f3F000000;\00max.NaN.f32 \09%f6, %f5, 0f00000000;\00st.global.f32 \09[%rd14+2048], %f6;\00$L__BB0_6:\00add.s32 \09%r13, %r6, -1024;\00min.s32 \09%r14, %r13, 512;\00setp.ge.s32 \09%p4, %r1, %r14;\00@%p4 bra \09$L__BB0_8;\00ld.global.f32 \09%f7, [%rd4+4096];\00mul.f32 \09%f8, %f7, 0f3F000000;\00max.NaN.f32 \09%f9, %f8, 0f00000000;\00st.global.f32 \09[%rd3+4096], %f9;\00$L__BB0_8:\00add.s32 \09%r16, %r6, -1536;\00min.s32 \09%r17, %r16, 512;\00setp.ge.s32 \09%p5, %r1, %r17;\00@%p5 bra \09$L__BB0_10;\00add.s32 \09%r18, %r1, %r2;\00add.s32 \09%r19, %r18, 1536;\00mul.wide.s32 \09%rd16, %r19, 4;\00add.s64 \09%rd5, %rd1, %rd16;\00add.s64 \09%rd6, %rd2, %rd16;\00ld.global.f32 \09%f10, [%rd6];\00mul.f32 \09%f11, %f10, 0f3F000000;\00max.NaN.f32 \09%f12, %f11, 0f00000000;\00st.global.f32 \09[%rd5], %f12;\00$L__BB0_10:\00ret;\00\00}\00\00\00\00\04/\08\00\08\00\00\00\15\00\00\00\04#\08\00\08\00\00\00\00\00\00\00\04\12\08\00\08\00\00\00\00\00\00\00\04\11\08\00\08\00\00\00\00\00\00\00\047\04\00z\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01\1C\00\03\19\1C\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0\11\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0\11\00\03\1B\FF\00\04\1C\0C\00`\02\00\00\10\05\00\00\90\05\00\00\04\1E\04\00\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056\1D\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00D\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\C4\0F\00\19y\05\00\00\00\00\00\00%\00\00\00\22\0E\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19y\00\00\00\00\00\00\00!\00\00\00b\0E\00\0Cz\00\05\00X\00\00pb\F0\03\00\E2\1F\04$x\05\05\00\08\00\00\FF\00\8E\07\00\D8\0F\00G\09\00\00\00\02\00\00\00\00\80\03\00\EA\0F\00\10r\12\05\00\00\00\00\FF\E0\F1\07\00\E2/\0C$t\09\FF\04\00\00\00\FF\00\8E\07\00\E2\0F\00\12r\02\05\00\00\00\00\FF\FC\8E\07\00\E4\0F\04\11r\03\05\FF\00\00\00\FF\0E\0F\00\00\E2\0F\00$x\0E\12\04\00\00\00\FF\00\8E\07\00\E2\0F\00\10x\00\00\00\06\00\00\05\E0\FF\07\00\E2\0F\00%v\04\02\00Z\00\00\09\02\8E\07\00\E2\0F\08\19x\12\12\02\00\00\00\03\02\01\00\00\E4\0F\00\10z\0A\0E\00Z\00\00\FF\E0\F1\07\00\E2\0F\00%v\06\00\00Z\00\00\09\02\8E\07\00\E2\0F\00\81y\0C\04\04\00\00\00\00\19\1E\0C\00\A4\0E\00\10z\0B\12\00[\00\00\FF\E4\7F\00\00\C4\0F\00\81y\0D\04\04\00\10\00\00\19\1E\0C\00\E8\0E\00\81y\0A\0A\04\00\08\00\00\19\1E\0C\00(\0F\00\81y\06\06\04\00\00\00\00\19\1E\0C\00b\0F\00%v\02\02\00\\\00\00\09\02\8E\07\00\C8\0F\00%v\08\00\00\\\00\00\09\02\8E\07\00\E2\0F\00 x\00\0C\00\00\00?\00\00@\00\00\E2O\00\10z\0C\0E\00\\\00\00\FF\E0\F1\07\00\E2\0F\00 x\0F\0D\00\00\00?\00\00@\00\00\E2\8F\00 x\0E\0A\00\00\00?\00\00@\00\00\E4\0F\01\09r\11\FF\00\00\00\00\00\00\82\07\00\E2\0F\00 x\10\06\00\00\00?\00\00@\00\00\E2\0F\02\10z\0D\12\00]\00\00\FF\E4\7F\00\00\E4\0F\00\09r\05\FF\0E\00\00\00\00\00\82\07\00\C4\0F\00\09r\0F\FF\0F\00\00\00\00\00\82\07\00\E4\0F\00\09r\07\FF\10\00\00\00\00\00\82\07\00\E2\0F\00\86y\00\02\11\00\00\00\04\19\10\0C\00\E8\0F\00\86y\00\0C\05\00\08\00\04\19\10\0C\00\E8\0F\00\86y\00\02\0F\00\10\00\04\19\10\0C\00\E8\0F\00\86y\00\08\07\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\10z\04\05\00^\00\00\FF\E1\FF\07\00\E2/\04$t\0F\FF\04\00\00\00\FF\00\8E\07\00\E2\0F\00\12r\0A\05\00\00\00\00\FF\FC\8E\07\00\C4\0F\00\17x\03\04\00\02\00\00\00\02\80\03\00\C8\0F\00\0Cr\00\00\03\00\00\00pb\F4\03\00\E2\0F\00%v\02\0A\00Z\00\00\0F\02\8E\07\00\D8\0F\00\81\A9\06\02\04\00\00\00\00\19\1E\0C\00\A2\0E\00\10x\07\04\00\FE\FF\FF\FF\E0\FF\07\00\E2\0F\04Ey\00\00\A0\01\00\00\00\00\80\03\00\E2\0F\00\10x\08\04\00\FC\FF\FF\FF\E0\FF\07\00\E4\0F\04\10x\04\04\00\FA\FF\FF\FF\E0\FF\07\00\E4\0F\00\17x\07\07\00\02\00\00\00\02\80\03\00\E4\0F\00\17x\0B\04\00\02\00\00\00\02\80\03\00\E4\0F\00\17x\09\08\00\02\00\00\00\02\80\03\00\C4\0F\00\0Cr\00\00\07\00\00\00pb\F6\03\00\E4\0F\04\0Cr\00\00\09\00\00\00pb\F0\03\00\E4\0F\04\0Cr\00\00\0B\00\00\00pb\F2\03\00\E2\0F\00 \A8\04\06\00\00\00?\00\00@\00\00\E2O\00%v\06\0A\00\\\00\00\0F\02\8E\07\00\C8\0F\00\09\A2\09\FF\04\00\00\00\00\00\82\07\00\CA\0F\00\86\A9\00\06\09\00\00\00\04\19\10\0C\00\E2\01\00G9\00\00\C0\00\00\00\00\00\80\03\00\EA\0F\00\10r\04\05\00\00\00\00\FF\E0\F5\07\00\C8\0F\00\11r\09\05\FF\00\00\00\FF\0E\0F\01\00\E2\1F\00$x\0A\04\04\00\00\00\FF\00\8E\07\00\C6\0F\00\19x\0B\04\02\00\00\00\09\02\01\00\00\E4\0F\00\10z\08\0A\00Z\00\00\FF\E0\F5\07\00\C8\0F\00\10z\09\0B\00[\00\00\FF\E4\7F\01\00\CA\0F\00\81y\08\08\04\00\08\00\00\19\1E\0C\00\A2\0E\00\10z\0A\0A\00\\\00\00\FF\E0\F5\07\00\C8\0F\00\10z\0B\0B\00]\00\00\FF\E4\7F\01\00\E2\0F\00 x\04\08\00\00\00?\00\00@\00\00\CAO\00\09r\0D\FF\04\00\00\00\00\00\82\07\00\CA\0F\00\86y\00\0A\0D\00\08\00\04\19\10\0C\00\E4\01\00Ay\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Ey\00\00`\00\00\00\00\00\80\03\00\E2\0F\00G\09\00\00@\00\00\00\00\00\80\03\00\EA\0F\00\81y\02\02\04\00\10\00\00\19\1E\0C\00\A4\0E\00 x\04\02\00\00\00?\00\00@\00\00\CAO\00\09r\09\FF\04\00\00\00\00\00\82\07\00\CA\1F\00\86y\00\06\09\00\10\00\04\19\10\0C\00\E4\01\00Ay\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00M\19\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\10x\02\00\00\06\00\00\05\E0\FF\07\00\CA\0F\00%v\04\02\00Z\00\00\0F\02\8E\07\00\CC\0F\00\81y\04\04\04\00\00\00\00\19\1E\0C\00\A2\0E\00%v\02\02\00\\\00\00\0F\02\8E\07\00\E2\0F\00 x\00\04\00\00\00?\00\00@\00\00\CAO\00\09r\07\FF\00\00\00\00\00\00\82\07\00\CA\1F\00\86y\00\02\07\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B1\01\00\00\00\00\00\00\98\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\03\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\E8\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\04\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F5\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\98\04\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\01\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B2\04\00\00\00\00\00\00\E6\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00-\01\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\98\05\00\00\00\00\00\00}\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\007\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\12\00\00\00\00\00\000\00\00\00\00\00\00\00\05\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00f\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00H\12\00\00\00\00\00\00x\00\00\00\00\00\00\00\05\00\00\00\0E\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00b\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C0\12\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\15\01\00\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\12\00\00\00\00\00\00\10\00\00\00\00\00\00\00\05\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00?\01\00\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E0\12\00\00\00\00\00\00\10\00\00\00\00\00\00\00\05\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\BA\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F0\12\00\00\00\00\00\00|\01\00\00\00\00\00\00\00\00\00\00\0E\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\14\00\00\00\00\00\00\80\06\00\00\00\00\00\00\05\00\00\00\08\00\00\15\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\C0\1E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\F0\12\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\C0\1E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00"} {
    llvm.func @predict_online_5603970_0_kernel(%arg0: i32, %arg1: !llvm.ptr<f32> {llvm.align = 16 : index}, %arg2: !llvm.ptr<f32> {llvm.align = 16 : index, llvm.noalias}, %arg3: i32) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.constant(-1 : index) : i32
      %1 = llvm.mlir.constant(2048 : index) : i32
      %2 = llvm.mlir.constant(512 : index) : i32
      %3 = llvm.mlir.constant(1024 : index) : i32
      %4 = llvm.mlir.constant(1536 : index) : i32
      %5 = llvm.mlir.constant(5.000000e-01 : f32) : f32
      %6 = llvm.mlir.constant(0.000000e+00 : f32) : f32
      %7 = nvvm.read.ptx.sreg.ctaid.x : i32
      %8 = nvvm.read.ptx.sreg.tid.x : i32
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %9 = llvm.mul %7, %1  : i32
      %10 = llvm.icmp "slt" %7, %arg0 : i32
      llvm.cond_br %10, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %11 = llvm.add %8, %2  : i32
      %12 = llvm.add %8, %3  : i32
      %13 = llvm.add %8, %4  : i32
      %14 = llvm.add %8, %9  : i32
      %15 = llvm.add %11, %9  : i32
      %16 = llvm.add %12, %9  : i32
      %17 = llvm.add %13, %9  : i32
      %18 = llvm.getelementptr %arg1[%14] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %19 = llvm.load %18 : !llvm.ptr<f32>
      %20 = llvm.fmul %19, %5  : f32
      %21 = llvm.getelementptr %arg1[%15] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %22 = llvm.load %21 : !llvm.ptr<f32>
      %23 = llvm.fmul %22, %5  : f32
      %24 = llvm.getelementptr %arg1[%16] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %25 = llvm.load %24 : !llvm.ptr<f32>
      %26 = llvm.fmul %25, %5  : f32
      %27 = llvm.getelementptr %arg1[%17] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %28 = llvm.load %27 : !llvm.ptr<f32>
      %29 = llvm.fmul %28, %5  : f32
      %30 = "llvm.intr.maximum"(%20, %6) : (f32, f32) -> f32
      %31 = llvm.getelementptr %arg2[%14] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %30, %31 : !llvm.ptr<f32>
      %32 = "llvm.intr.maximum"(%23, %6) : (f32, f32) -> f32
      %33 = llvm.getelementptr %arg2[%15] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %32, %33 : !llvm.ptr<f32>
      %34 = "llvm.intr.maximum"(%26, %6) : (f32, f32) -> f32
      %35 = llvm.getelementptr %arg2[%16] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %34, %35 : !llvm.ptr<f32>
      %36 = "llvm.intr.maximum"(%29, %6) : (f32, f32) -> f32
      %37 = llvm.getelementptr %arg2[%17] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %36, %37 : !llvm.ptr<f32>
      llvm.br ^bb12
    ^bb3:  // pred: ^bb1
      %38 = llvm.mul %9, %0  : i32
      %39 = llvm.add %38, %arg3  : i32
      %40 = llvm.icmp "slt" %39, %2 : i32
      %41 = llvm.select %40, %39, %2 : i1, i32
      %42 = llvm.icmp "slt" %8, %41 : i32
      llvm.cond_br %42, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %43 = llvm.add %8, %9  : i32
      %44 = llvm.getelementptr %arg1[%43] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %45 = llvm.load %44 : !llvm.ptr<f32>
      %46 = llvm.fmul %45, %5  : f32
      %47 = "llvm.intr.maximum"(%46, %6) : (f32, f32) -> f32
      %48 = llvm.getelementptr %arg2[%43] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %47, %48 : !llvm.ptr<f32>
      llvm.br ^bb5
    ^bb5:  // 2 preds: ^bb3, ^bb4
      %49 = llvm.add %9, %2  : i32
      %50 = llvm.mul %49, %0  : i32
      %51 = llvm.add %50, %arg3  : i32
      %52 = llvm.icmp "slt" %51, %2 : i32
      %53 = llvm.select %52, %51, %2 : i1, i32
      %54 = llvm.add %8, %2  : i32
      %55 = llvm.icmp "slt" %8, %53 : i32
      llvm.cond_br %55, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %56 = llvm.add %54, %9  : i32
      %57 = llvm.getelementptr %arg1[%56] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %58 = llvm.load %57 : !llvm.ptr<f32>
      %59 = llvm.fmul %58, %5  : f32
      %60 = "llvm.intr.maximum"(%59, %6) : (f32, f32) -> f32
      %61 = llvm.getelementptr %arg2[%56] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %60, %61 : !llvm.ptr<f32>
      llvm.br ^bb7
    ^bb7:  // 2 preds: ^bb5, ^bb6
      %62 = llvm.add %9, %3  : i32
      %63 = llvm.mul %62, %0  : i32
      %64 = llvm.add %63, %arg3  : i32
      %65 = llvm.icmp "slt" %64, %2 : i32
      %66 = llvm.select %65, %64, %2 : i1, i32
      %67 = llvm.add %8, %3  : i32
      %68 = llvm.icmp "slt" %8, %66 : i32
      llvm.cond_br %68, ^bb8, ^bb9
    ^bb8:  // pred: ^bb7
      %69 = llvm.add %67, %9  : i32
      %70 = llvm.getelementptr %arg1[%69] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %71 = llvm.load %70 : !llvm.ptr<f32>
      %72 = llvm.fmul %71, %5  : f32
      %73 = "llvm.intr.maximum"(%72, %6) : (f32, f32) -> f32
      %74 = llvm.getelementptr %arg2[%69] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %73, %74 : !llvm.ptr<f32>
      llvm.br ^bb9
    ^bb9:  // 2 preds: ^bb7, ^bb8
      %75 = llvm.add %9, %4  : i32
      %76 = llvm.mul %75, %0  : i32
      %77 = llvm.add %76, %arg3  : i32
      %78 = llvm.icmp "slt" %77, %2 : i32
      %79 = llvm.select %78, %77, %2 : i1, i32
      %80 = llvm.add %8, %4  : i32
      %81 = llvm.icmp "slt" %8, %79 : i32
      llvm.cond_br %81, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %82 = llvm.add %80, %9  : i32
      %83 = llvm.getelementptr %arg1[%82] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      %84 = llvm.load %83 : !llvm.ptr<f32>
      %85 = llvm.fmul %84, %5  : f32
      %86 = "llvm.intr.maximum"(%85, %6) : (f32, f32) -> f32
      %87 = llvm.getelementptr %arg2[%82] : (!llvm.ptr<f32>, i32) -> !llvm.ptr<f32>
      llvm.store %86, %87 : !llvm.ptr<f32>
      llvm.br ^bb11
    ^bb11:  // 2 preds: ^bb9, ^bb10
      llvm.br ^bb12
    ^bb12:  // 2 preds: ^bb2, ^bb11
      llvm.return
    }
  }
}

module attributes {gpu.container_module} {
  llvm.func @_sony_ciface_launch_kernel(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, i64, i64, i64, i64, i64, !llvm.ptr<ptr<i8>>, i32, i32)
  llvm.mlir.global external constant @predict_online_5603970_0_kernel_blob("\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00z\00\00\00\00\00\00\00\00\00\00\00\C0\1E\00\00\00\00\00\00\00\1B\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0F\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.uft.entry\00.nv.info\00.text.predict_online_5603970_0_kernel\00.nv.info.predict_online_5603970_0_kernel\00.nv.shared.predict_online_5603970_0_kernel\00.nv.constant0.predict_online_5603970_0_kernel\00.debug_frame\00.debug_line\00.nv_debug_line_sass\00.rel.nv_debug_line_sass\00.nv_debug_ptx_txt\00.rel.debug_frame\00.rela.debug_frame\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.uft.entry\00.nv.info\00predict_online_5603970_0_kernel\00.text.predict_online_5603970_0_kernel\00.nv.info.predict_online_5603970_0_kernel\00.nv.shared.predict_online_5603970_0_kernel\00.nv.constant0.predict_online_5603970_0_kernel\00_param\00.debug_frame\00.debug_line\00.nv_debug_line_sass\00.rel.nv_debug_line_sass\00.nv_debug_ptx_txt\00.rel.debug_frame\00.rela.debug_frame\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00`\00\00\00\03\00\0E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DA\00\00\00\03\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\1C\01\00\00\03\00\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\01\00\00\03\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00T\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\89\01\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\12\10\0E\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\06\00\00\00\00\00\00\04\04\00\00\00\04\18\00\00\00\0C\81\80\80(\00\04H\01\00\00\00\00\00\00\00\00\00\16\00\00\00\02\00\10\00\00\00\01\01\FB\0E\0A\00\01\01\01\01\00\00\00\01\00\00\E2\00\00\00\02\00\10\00\00\00\01\01\FB\0E\0A\00\01\01\01\01\00\00\00\01\00\00\00\09\02\00\00\00\00\00\00\00\00\04\00\03\10\01\03\0B\02\10\01\03u\02\10\01\03\0C\02\10\01\F1\EE\F1\03\0A\02\10\01\EA\EC\F7\F0\03x\02\10\01\F1\F5\F0\F5\03t\02\10\01\F5\F2\ED\F5\F2\F7\03i\02\10\01\03\12\02\10\01\03w\02\10\01\ED\F6\EE\F4\EE\F2\F1\03z\02\10\01\F2\F1\F2\F0\F2\F4\ED\ED\F0\F3\F1\F4\ED\03\11\02\10\01\03\09\02\10\01\03i\02\10\01\03\18\02\10\01\03w\02\10\01\03r\02\10\01\03\0F\02\10\01\03\09\02\10\01\03b\02\10\01\EB\F4\F0\F4\F2\03\01\02 \01\03\02\02 \01\03\01\02 \01\ED\03\03\02 \01\F0\F0\03\05\020\01\F0\F0\F0\F0\03\05\02 \01\F1\F2\F0\ED\F2\F0\F0\F1\02\F0\01\00\01\01\00\00\00\00.version 7.1\00.target sm_86\00.address_size 64\00\00\00\00.visible .entry predict_online_5603970_0_kernel(\00.param .u32 predict_online_5603970_0_kernel_param_0,\00.param .u64 predict_online_5603970_0_kernel_param_1,\00.param .u64 predict_online_5603970_0_kernel_param_2,\00.param .u32 predict_online_5603970_0_kernel_param_3\00)\00{\00.reg .pred \09%p<6>;\00.reg .b32 \09%r<23>;\00.reg .f32 \09%f<25>;\00.reg .b64 \09%rd<29>;\00\00ld.param.u32 \09%r4, [predict_online_5603970_0_kernel_param_0];\00ld.param.u64 \09%rd7, [predict_online_5603970_0_kernel_param_2];\00cvta.to.global.u64 \09%rd1, %rd7;\00ld.param.u64 \09%rd8, [predict_online_5603970_0_kernel_param_1];\00cvta.to.global.u64 \09%rd2, %rd8;\00mov.u32 \09%r5, %ctaid.x;\00mov.u32 \09%r1, %tid.x;\00shl.b32 \09%r2, %r5, 11;\00setp.ge.s32 \09%p1, %r5, %r4;\00@%p1 bra \09$L__BB0_2;\00add.s32 \09%r20, %r1, %r2;\00or.b32  \09%r21, %r2, %r1;\00add.s32 \09%r22, %r20, 1536;\00mul.wide.s32 \09%rd17, %r21, 4;\00add.s64 \09%rd18, %rd2, %rd17;\00ld.global.f32 \09%f13, [%rd18];\00mul.f32 \09%f14, %f13, 0f3F000000;\00cvt.s64.s32 \09%rd19, %r2;\00cvt.u64.u32 \09%rd20, %r1;\00add.s64 \09%rd21, %rd20, %rd19;\00shl.b64 \09%rd22, %rd21, 2;\00add.s64 \09%rd23, %rd2, %rd22;\00ld.global.f32 \09%f15, [%rd23+2048];\00mul.f32 \09%f16, %f15, 0f3F000000;\00ld.global.f32 \09%f17, [%rd18+4096];\00mul.f32 \09%f18, %f17, 0f3F000000;\00mul.wide.s32 \09%rd24, %r22, 4;\00add.s64 \09%rd25, %rd2, %rd24;\00ld.global.f32 \09%f19, [%rd25];\00mul.f32 \09%f20, %f19, 0f3F000000;\00max.NaN.f32 \09%f21, %f14, 0f00000000;\00add.s64 \09%rd26, %rd1, %rd17;\00st.global.f32 \09[%rd26], %f21;\00max.NaN.f32 \09%f22, %f16, 0f00000000;\00add.s64 \09%rd27, %rd1, %rd22;\00st.global.f32 \09[%rd27+2048], %f22;\00max.NaN.f32 \09%f23, %f18, 0f00000000;\00st.global.f32 \09[%rd26+4096], %f23;\00max.NaN.f32 \09%f24, %f20, 0f00000000;\00add.s64 \09%rd28, %rd1, %rd24;\00st.global.f32 \09[%rd28], %f24;\00bra.uni \09$L__BB0_10;\00$L__BB0_2:\00ld.param.u32 \09%r3, [predict_online_5603970_0_kernel_param_3];\00sub.s32 \09%r6, %r3, %r2;\00min.s32 \09%r7, %r6, 512;\00setp.ge.s32 \09%p2, %r1, %r7;\00or.b32  \09%r8, %r2, %r1;\00mul.wide.s32 \09%rd9, %r8, 4;\00add.s64 \09%rd3, %rd1, %rd9;\00add.s64 \09%rd4, %rd2, %rd9;\00@%p2 bra \09$L__BB0_4;\00ld.global.f32 \09%f1, [%rd4];\00mul.f32 \09%f2, %f1, 0f3F000000;\00max.NaN.f32 \09%f3, %f2, 0f00000000;\00st.global.f32 \09[%rd3], %f3;\00$L__BB0_4:\00add.s32 \09%r10, %r6, -512;\00min.s32 \09%r11, %r10, 512;\00setp.ge.s32 \09%p3, %r1, %r11;\00@%p3 bra \09$L__BB0_6;\00cvt.s64.s32 \09%rd10, %r2;\00cvt.u64.u32 \09%rd11, %r1;\00add.s64 \09%rd12, %rd11, %rd10;\00shl.b64 \09%rd13, %rd12, 2;\00add.s64 \09%rd14, %rd1, %rd13;\00add.s64 \09%rd15, %rd2, %rd13;\00ld.global.f32 \09%f4, [%rd15+2048];\00mul.f32 \09%f5, %f4, 0f3F000000;\00max.NaN.f32 \09%f6, %f5, 0f00000000;\00st.global.f32 \09[%rd14+2048], %f6;\00$L__BB0_6:\00add.s32 \09%r13, %r6, -1024;\00min.s32 \09%r14, %r13, 512;\00setp.ge.s32 \09%p4, %r1, %r14;\00@%p4 bra \09$L__BB0_8;\00ld.global.f32 \09%f7, [%rd4+4096];\00mul.f32 \09%f8, %f7, 0f3F000000;\00max.NaN.f32 \09%f9, %f8, 0f00000000;\00st.global.f32 \09[%rd3+4096], %f9;\00$L__BB0_8:\00add.s32 \09%r16, %r6, -1536;\00min.s32 \09%r17, %r16, 512;\00setp.ge.s32 \09%p5, %r1, %r17;\00@%p5 bra \09$L__BB0_10;\00add.s32 \09%r18, %r1, %r2;\00add.s32 \09%r19, %r18, 1536;\00mul.wide.s32 \09%rd16, %r19, 4;\00add.s64 \09%rd5, %rd1, %rd16;\00add.s64 \09%rd6, %rd2, %rd16;\00ld.global.f32 \09%f10, [%rd6];\00mul.f32 \09%f11, %f10, 0f3F000000;\00max.NaN.f32 \09%f12, %f11, 0f00000000;\00st.global.f32 \09[%rd5], %f12;\00$L__BB0_10:\00ret;\00\00}\00\00\00\00\04/\08\00\08\00\00\00\15\00\00\00\04#\08\00\08\00\00\00\00\00\00\00\04\12\08\00\08\00\00\00\00\00\00\00\04\11\08\00\08\00\00\00\00\00\00\00\047\04\00z\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01\1C\00\03\19\1C\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0\11\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0\11\00\03\1B\FF\00\04\1C\0C\00`\02\00\00\10\05\00\00\90\05\00\00\04\1E\04\00\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056\1D\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00D\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\C4\0F\00\19y\05\00\00\00\00\00\00%\00\00\00\22\0E\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19y\00\00\00\00\00\00\00!\00\00\00b\0E\00\0Cz\00\05\00X\00\00pb\F0\03\00\E2\1F\04$x\05\05\00\08\00\00\FF\00\8E\07\00\D8\0F\00G\09\00\00\00\02\00\00\00\00\80\03\00\EA\0F\00\10r\12\05\00\00\00\00\FF\E0\F1\07\00\E2/\0C$t\09\FF\04\00\00\00\FF\00\8E\07\00\E2\0F\00\12r\02\05\00\00\00\00\FF\FC\8E\07\00\E4\0F\04\11r\03\05\FF\00\00\00\FF\0E\0F\00\00\E2\0F\00$x\0E\12\04\00\00\00\FF\00\8E\07\00\E2\0F\00\10x\00\00\00\06\00\00\05\E0\FF\07\00\E2\0F\00%v\04\02\00Z\00\00\09\02\8E\07\00\E2\0F\08\19x\12\12\02\00\00\00\03\02\01\00\00\E4\0F\00\10z\0A\0E\00Z\00\00\FF\E0\F1\07\00\E2\0F\00%v\06\00\00Z\00\00\09\02\8E\07\00\E2\0F\00\81y\0C\04\04\00\00\00\00\19\1E\0C\00\A4\0E\00\10z\0B\12\00[\00\00\FF\E4\7F\00\00\C4\0F\00\81y\0D\04\04\00\10\00\00\19\1E\0C\00\E8\0E\00\81y\0A\0A\04\00\08\00\00\19\1E\0C\00(\0F\00\81y\06\06\04\00\00\00\00\19\1E\0C\00b\0F\00%v\02\02\00\\\00\00\09\02\8E\07\00\C8\0F\00%v\08\00\00\\\00\00\09\02\8E\07\00\E2\0F\00 x\00\0C\00\00\00?\00\00@\00\00\E2O\00\10z\0C\0E\00\\\00\00\FF\E0\F1\07\00\E2\0F\00 x\0F\0D\00\00\00?\00\00@\00\00\E2\8F\00 x\0E\0A\00\00\00?\00\00@\00\00\E4\0F\01\09r\11\FF\00\00\00\00\00\00\82\07\00\E2\0F\00 x\10\06\00\00\00?\00\00@\00\00\E2\0F\02\10z\0D\12\00]\00\00\FF\E4\7F\00\00\E4\0F\00\09r\05\FF\0E\00\00\00\00\00\82\07\00\C4\0F\00\09r\0F\FF\0F\00\00\00\00\00\82\07\00\E4\0F\00\09r\07\FF\10\00\00\00\00\00\82\07\00\E2\0F\00\86y\00\02\11\00\00\00\04\19\10\0C\00\E8\0F\00\86y\00\0C\05\00\08\00\04\19\10\0C\00\E8\0F\00\86y\00\02\0F\00\10\00\04\19\10\0C\00\E8\0F\00\86y\00\08\07\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\10z\04\05\00^\00\00\FF\E1\FF\07\00\E2/\04$t\0F\FF\04\00\00\00\FF\00\8E\07\00\E2\0F\00\12r\0A\05\00\00\00\00\FF\FC\8E\07\00\C4\0F\00\17x\03\04\00\02\00\00\00\02\80\03\00\C8\0F\00\0Cr\00\00\03\00\00\00pb\F4\03\00\E2\0F\00%v\02\0A\00Z\00\00\0F\02\8E\07\00\D8\0F\00\81\A9\06\02\04\00\00\00\00\19\1E\0C\00\A2\0E\00\10x\07\04\00\FE\FF\FF\FF\E0\FF\07\00\E2\0F\04Ey\00\00\A0\01\00\00\00\00\80\03\00\E2\0F\00\10x\08\04\00\FC\FF\FF\FF\E0\FF\07\00\E4\0F\04\10x\04\04\00\FA\FF\FF\FF\E0\FF\07\00\E4\0F\00\17x\07\07\00\02\00\00\00\02\80\03\00\E4\0F\00\17x\0B\04\00\02\00\00\00\02\80\03\00\E4\0F\00\17x\09\08\00\02\00\00\00\02\80\03\00\C4\0F\00\0Cr\00\00\07\00\00\00pb\F6\03\00\E4\0F\04\0Cr\00\00\09\00\00\00pb\F0\03\00\E4\0F\04\0Cr\00\00\0B\00\00\00pb\F2\03\00\E2\0F\00 \A8\04\06\00\00\00?\00\00@\00\00\E2O\00%v\06\0A\00\\\00\00\0F\02\8E\07\00\C8\0F\00\09\A2\09\FF\04\00\00\00\00\00\82\07\00\CA\0F\00\86\A9\00\06\09\00\00\00\04\19\10\0C\00\E2\01\00G9\00\00\C0\00\00\00\00\00\80\03\00\EA\0F\00\10r\04\05\00\00\00\00\FF\E0\F5\07\00\C8\0F\00\11r\09\05\FF\00\00\00\FF\0E\0F\01\00\E2\1F\00$x\0A\04\04\00\00\00\FF\00\8E\07\00\C6\0F\00\19x\0B\04\02\00\00\00\09\02\01\00\00\E4\0F\00\10z\08\0A\00Z\00\00\FF\E0\F5\07\00\C8\0F\00\10z\09\0B\00[\00\00\FF\E4\7F\01\00\CA\0F\00\81y\08\08\04\00\08\00\00\19\1E\0C\00\A2\0E\00\10z\0A\0A\00\\\00\00\FF\E0\F5\07\00\C8\0F\00\10z\0B\0B\00]\00\00\FF\E4\7F\01\00\E2\0F\00 x\04\08\00\00\00?\00\00@\00\00\CAO\00\09r\0D\FF\04\00\00\00\00\00\82\07\00\CA\0F\00\86y\00\0A\0D\00\08\00\04\19\10\0C\00\E4\01\00Ay\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Ey\00\00`\00\00\00\00\00\80\03\00\E2\0F\00G\09\00\00@\00\00\00\00\00\80\03\00\EA\0F\00\81y\02\02\04\00\10\00\00\19\1E\0C\00\A4\0E\00 x\04\02\00\00\00?\00\00@\00\00\CAO\00\09r\09\FF\04\00\00\00\00\00\82\07\00\CA\1F\00\86y\00\06\09\00\10\00\04\19\10\0C\00\E4\01\00Ay\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00M\19\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\10x\02\00\00\06\00\00\05\E0\FF\07\00\CA\0F\00%v\04\02\00Z\00\00\0F\02\8E\07\00\CC\0F\00\81y\04\04\04\00\00\00\00\19\1E\0C\00\A2\0E\00%v\02\02\00\\\00\00\0F\02\8E\07\00\E2\0F\00 x\00\04\00\00\00?\00\00@\00\00\CAO\00\09r\07\FF\00\00\00\00\00\00\82\07\00\CA\1F\00\86y\00\02\07\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B1\01\00\00\00\00\00\00\98\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00P\03\00\00\00\00\00\00\D8\00\00\00\00\00\00\00\02\00\00\00\08\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\E8\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\04\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F5\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\98\04\00\00\00\00\00\00\1A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\01\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\B2\04\00\00\00\00\00\00\E6\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00-\01\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\98\05\00\00\00\00\00\00}\0C\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\007\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\12\00\00\00\00\00\000\00\00\00\00\00\00\00\05\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00f\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00H\12\00\00\00\00\00\00x\00\00\00\00\00\00\00\05\00\00\00\0E\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00b\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\C0\12\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\15\01\00\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\12\00\00\00\00\00\00\10\00\00\00\00\00\00\00\05\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00?\01\00\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E0\12\00\00\00\00\00\00\10\00\00\00\00\00\00\00\05\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\BA\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F0\12\00\00\00\00\00\00|\01\00\00\00\00\00\00\00\00\00\00\0E\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\14\00\00\00\00\00\00\80\06\00\00\00\00\00\00\05\00\00\00\08\00\00\15\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\C0\1E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\F0\12\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\10\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\C0\1E\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00") {addr_space = 0 : i32}
  llvm.func @_sony_ciface_report_error(!llvm.ptr<i8>, i32, !llvm.ptr<i8>)
  llvm.mlir.global internal constant @error_message_18271215307014969330("failed to allocate memory\00") {addr_space = 0 : i32}
  llvm.func @_sony_ciface_alloc(!llvm.ptr<i8>, i64, i64, i32, i32, !llvm.ptr<i32>) -> !llvm.ptr<i8>
  llvm.func @predict_online_5603970_0(%arg0: !llvm.ptr<i8> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: !llvm.ptr<f32>, %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr<i8>) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %0 = llvm.mlir.constant(2048 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(512 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(8 : index) : i64
    %5 = llvm.mlir.constant(1024 : index) : i64
    %6 = llvm.mlir.constant(8192 : index) : i64
    %7 = llvm.mlir.constant(-1 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(false) : i1
    %10 = llvm.mlir.constant(-1 : index) : i64
    %11 = llvm.mlir.constant(true) : i1
    %12 = llvm.mlir.constant(1 : i32) : i32
    %13 = llvm.mlir.constant(4 : i32) : i32
    %14 = llvm.mlir.constant(8 : i32) : i32
    %15 = llvm.bitcast %arg10 : !llvm.ptr<i8> to !llvm.ptr<ptr<i8>>
    %16 = llvm.mul %arg4, %4  : i64
    %17 = llvm.mul %16, %5  : i64
    %18 = llvm.mlir.null : !llvm.ptr<f32>
    %19 = llvm.getelementptr %18[1] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %20 = llvm.ptrtoint %19 : !llvm.ptr<f32> to i64
    %21 = llvm.mlir.null : !llvm.ptr<i32>
    %22 = llvm.call @_sony_ciface_alloc(%arg0, %17, %20, %7, %8, %21) : (!llvm.ptr<i8>, i64, i64, i32, i32, !llvm.ptr<i32>) -> !llvm.ptr<i8>
    %23 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
    %24 = llvm.bitcast %22 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %25 = llvm.insertvalue %24, %23[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %24, %25[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.insertvalue %3, %26[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %28 = llvm.insertvalue %5, %27[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %29 = llvm.insertvalue %1, %28[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = llvm.mul %1, %5  : i64
    %31 = llvm.insertvalue %4, %29[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %32 = llvm.insertvalue %30, %31[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %33 = llvm.mul %30, %4  : i64
    %34 = llvm.insertvalue %arg4, %32[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %35 = llvm.insertvalue %33, %34[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %36 = llvm.icmp "eq" %arg4, %3 : i64
    %37 = llvm.or %9, %36  : i1
    %38 = llvm.icmp "eq" %4, %3 : i64
    %39 = llvm.or %37, %38  : i1
    %40 = llvm.icmp "eq" %5, %3 : i64
    %41 = llvm.or %39, %40  : i1
    %42 = llvm.mlir.null : !llvm.ptr<i8>
    %43 = llvm.icmp "ne" %22, %42 : !llvm.ptr<i8>
    %44 = llvm.or %43, %41  : i1
    llvm.cond_br %44, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %45 = llvm.sdiv %17, %0  : i64
    %46 = llvm.select %11, %10, %1 : i1, i64
    %47 = llvm.add %46, %17  : i64
    %48 = llvm.sdiv %47, %0  : i64
    %49 = llvm.add %48, %1  : i64
    %50 = llvm.sub %3, %17  : i64
    %51 = llvm.sdiv %50, %0  : i64
    %52 = llvm.sub %3, %51  : i64
    %53 = llvm.icmp "slt" %17, %3 : i64
    %54 = llvm.icmp "sgt" %17, %3 : i64
    %55 = llvm.and %53, %9  : i1
    %56 = llvm.and %54, %11  : i1
    %57 = llvm.or %55, %56  : i1
    %58 = llvm.select %57, %49, %52 : i1, i64
    %59 = llvm.mlir.addressof @predict_online_5603970_0_kernel_blob : !llvm.ptr<array<8040 x i8>>
    %60 = llvm.getelementptr %59[0, 0] : (!llvm.ptr<array<8040 x i8>>) -> !llvm.ptr<i8>
    %61 = llvm.alloca %12 x !llvm.struct<"", (i64, ptr<f32>, ptr<f32>, i64)> : (i32) -> !llvm.ptr<struct<"", (i64, ptr<f32>, ptr<f32>, i64)>>
    %62 = llvm.alloca %13 x !llvm.ptr<i8> : (i32) -> !llvm.ptr<ptr<i8>>
    %63 = llvm.getelementptr %61[0, 0] : (!llvm.ptr<struct<"", (i64, ptr<f32>, ptr<f32>, i64)>>) -> !llvm.ptr<i64>
    llvm.store %45, %63 : !llvm.ptr<i64>
    %64 = llvm.bitcast %63 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %64, %62 : !llvm.ptr<ptr<i8>>
    %65 = llvm.getelementptr %61[0, 1] : (!llvm.ptr<struct<"", (i64, ptr<f32>, ptr<f32>, i64)>>) -> !llvm.ptr<ptr<f32>>
    llvm.store %arg2, %65 : !llvm.ptr<ptr<f32>>
    %66 = llvm.getelementptr %62[1] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %67 = llvm.bitcast %65 : !llvm.ptr<ptr<f32>> to !llvm.ptr<i8>
    llvm.store %67, %66 : !llvm.ptr<ptr<i8>>
    %68 = llvm.getelementptr %61[0, 2] : (!llvm.ptr<struct<"", (i64, ptr<f32>, ptr<f32>, i64)>>) -> !llvm.ptr<ptr<f32>>
    llvm.store %24, %68 : !llvm.ptr<ptr<f32>>
    %69 = llvm.getelementptr %62[2] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %70 = llvm.bitcast %68 : !llvm.ptr<ptr<f32>> to !llvm.ptr<i8>
    llvm.store %70, %69 : !llvm.ptr<ptr<i8>>
    %71 = llvm.getelementptr %61[0, 3] : (!llvm.ptr<struct<"", (i64, ptr<f32>, ptr<f32>, i64)>>) -> !llvm.ptr<i64>
    llvm.store %17, %71 : !llvm.ptr<i64>
    %72 = llvm.getelementptr %62[3] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %73 = llvm.bitcast %71 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %73, %72 : !llvm.ptr<ptr<i8>>
    %74 = llvm.load %15 : !llvm.ptr<ptr<i8>>
    llvm.call @_sony_ciface_launch_kernel(%arg0, %60, %74, %58, %1, %1, %2, %1, %1, %62, %8, %8) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, i64, i64, i64, i64, i64, !llvm.ptr<ptr<i8>>, i32, i32) -> ()
    llvm.return %35 : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
  ^bb2:  // pred: ^bb0
    %75 = llvm.mlir.addressof @error_message_18271215307014969330 : !llvm.ptr<array<26 x i8>>
    %76 = llvm.getelementptr %75[0, 0] : (!llvm.ptr<array<26 x i8>>) -> !llvm.ptr<i8>
    llvm.call @_sony_ciface_report_error(%arg0, %14, %76) : (!llvm.ptr<i8>, i32, !llvm.ptr<i8>) -> ()
    %77 = llvm.insertvalue %18, %23[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %78 = llvm.insertvalue %18, %77[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %79 = llvm.insertvalue %3, %78[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %80 = llvm.insertvalue %1, %79[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %81 = llvm.insertvalue %6, %80[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %82 = llvm.insertvalue %4, %81[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %83 = llvm.insertvalue %5, %82[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %84 = llvm.insertvalue %5, %83[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %85 = llvm.insertvalue %1, %84[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.return %85 : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
  }
  llvm.func @_sony_ciface_predict_online_5603970_0(%arg0: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>>, %arg1: !llvm.ptr<i8>, %arg2: !llvm.ptr<ptr<i8>>, %arg3: !llvm.ptr<i8>) {
    %0 = llvm.load %arg2 : !llvm.ptr<ptr<i8>>
    %1 = llvm.bitcast %0 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>>
    %2 = llvm.load %1 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>>
    %3 = llvm.extractvalue %2[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.extractvalue %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.extractvalue %2[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.extractvalue %2[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.extractvalue %2[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.extractvalue %2[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.extractvalue %2[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %11 = llvm.extractvalue %2[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.call @predict_online_5603970_0(%arg1, %3, %4, %5, %6, %7, %8, %9, %10, %11, %arg3) : (!llvm.ptr<i8>, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr<i8>) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
    llvm.store %12, %arg0 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>>
    llvm.return
  }
}

