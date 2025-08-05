# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.


class FlatQuantConfig:
    def __init__(self):
        # General Arguments
        self.seed = 0
        self.quant_by_quant = False
        self.diag_relu = True
        self.amp_dtype = "bfloat16"

        # Activation Quantization Arguments
        self.a_bits = 4
        self.a_groupsize = -1
        self.a_asym = False

        # Weight Quantization Arguments
        self.w_bits = 4
        self.w_groupsize = -1
        self.w_asym = False
        self.gptq = False
        self.gptq_mse = False
        self.percdamp = 0.01
        self.act_order = False

        # FlatQuant calibration Arguments
        self.epochs = 15
        self.nsamples = None
        self.cali_bsz = 1
        self.flat_lr = 5e-3
        self.cali_trans = True
        self.add_diag = True
        self.lwc = True
        self.lac = True
        self.diag_init = "sq_style"
        self.diag_alpha = 0.3
        self.warmup = False
        self.deactive_amp = False
        self.direct_inv = False

        self.quantize = (self.w_bits < 16) or (self.a_bits < 16)