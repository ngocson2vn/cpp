class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "f32[s0, 64, 64]", arg2_1: "f32[s0, 31, 64]"):
         # File: /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/inductor_bmm_fusion.py:49 in forward, code: tmp1 = torch.bmm(x, y)
        bmm: "f32[s0, 31, 64]" = torch.ops.aten.bmm.default(arg2_1, arg1_1);  arg2_1 = arg1_1 = None
        
         # File: /data00/home/son.nguyen/workspace/cpp/MLCompiler/pytorch/inductor_bmm_fusion.py:50 in forward, code: res = torch.sigmoid(tmp1)
        sigmoid: "f32[s0, 31, 64]" = torch.ops.aten.sigmoid.default(bmm);  bmm = None
        return (sigmoid,)
        