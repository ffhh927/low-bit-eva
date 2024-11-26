import torch
import qtensor.functional as F


class QTensor:
    def __init__(self, var, bits=32, name2qmap={}, code='', blocksize=2048, min_lowbit_size=4096):
        self.bits = bits

        if self.bits in [32, 16] or var.numel() < min_lowbit_size:
            if self.bits == 16:
                self.var = var.bfloat16()
                self.bits = 16
            else:
                self.var = var.float()
                self.bits = 32
        elif self.bits in [8, 4]:
            name2qmap[code] = name2qmap[code].to(var.device)
            self.name2qmap = name2qmap
            self.code = code
            self.blocksize = blocksize

            self.var_order = var.shape[0]
            self.var_dtype = var.dtype
            self.var, self.absmax = F.quantize_blockwise(var, code=self.name2qmap[self.code], order=self.var_order, blocksize=self.blocksize, bits=self.bits)
        else:
            raise ValueError(f'num of bits is not supported: {self.bits}')
    
    def quantize(self, var):
        if self.bits < 16:
            F.quantize_blockwise(var.contiguous(), code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, out=self.var, blocksize=self.blocksize, bits=self.bits)
        else:
            self.var = var.to(self.var.dtype)

    def dequantize(self):
        if self.bits < 16:
            return F.dequantize_blockwise(self.var, code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, outdtype=self.var_dtype, blocksize=self.blocksize, bits=self.bits)
        else:
            return self.var

    def set_device(self, device):
        if self.bits < 16:
            self.name2qmap[self.code] = self.name2qmap[self.code].to(device)
            self.absmax = self.absmax.to(device)
        self.var = self.var.to(device)


class QTensorDiagReal:
    def __init__(self, var, bits=32, name2qmap={}, code='', blocksize=2048, min_lowbit_size=4096):
        self.bits = bits

        if self.bits in [32, 16] or var.numel() < min_lowbit_size:
            if self.bits == 16:
                self.var = var.bfloat16()
                self.bits = 16
            else:
                self.var = var.float()
                self.bits = 32
        elif self.bits in [8, 4]:
            name2qmap[code] = name2qmap[code].to(var.device)
            self.name2qmap = name2qmap
            self.code = code
            self.blocksize = blocksize

            self.var_order = var.shape[0]
            self.var_dtype = var.dtype
            self.var, self.absmax, self.diag = F.quantize_blockwise_diagreal(var, code=self.name2qmap[self.code], order=self.var_order, blocksize=self.blocksize, bits=self.bits)
        else:
            raise ValueError(f'num of bits is not supported: {self.bits}')
    
    def quantize(self, var):
        if self.bits < 16:
            F.quantize_blockwise_diagreal(var.contiguous(), code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, diag=self.diag, out=self.var, blocksize=self.blocksize, bits=self.bits)
        else:
            self.var = var.to(self.var.dtype)

    def dequantize(self):
        if self.bits < 16:
            return F.dequantize_blockwise_diagreal(self.var, code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, diag=self.diag, outdtype=self.var_dtype, blocksize=self.blocksize, bits=self.bits)
        else:
            return self.var

    def set_device(self, device):
        if self.bits < 16:
            self.name2qmap[self.code] = self.name2qmap[self.code].to(device)
            self.absmax = self.absmax.to(device)
            self.diag = self.diag.to(device)
        self.var = self.var.to(device)


class QTensorSVDFast:
    def __init__(self, var, bits=32, name2qmap={}, code='', blocksize=2048, min_lowbit_size=4096, rect_t1=1, rect_t2=4):
        self.bits = bits
        self.rect_t1 = rect_t1
        self.rect_t2 = rect_t2

        Vt = torch.eye(var.shape[0], device=var.device)
        self.Svalue = var[0][0] * Vt.diag()

        if self.bits in [32, 16] or var.numel() < min_lowbit_size:
            self.var = Vt
            self.bits = 32
        elif self.bits in [8, 4]:
            name2qmap[code] = name2qmap[code].to(var.device)
            self.name2qmap = name2qmap
            self.code = code
            self.blocksize = blocksize

            self.var_order = var.shape[0]
            self.var_dtype = Vt.dtype
            self.var, self.absmax = F.quantize_blockwise(Vt, code=self.name2qmap[self.code], order=self.var_order, blocksize=self.blocksize, bits=self.bits)
        else:
            raise ValueError(f'num of bits is not supported: {self.bits}')
    
    def quantize(self, var, Vt=None):
        V, _ = torch.linalg.qr(var.float() @ Vt.T.float())
        self.Svalue = (V.T @ var.float() @ V).diag()

        if self.bits < 16:
            F.quantize_blockwise(V.T.contiguous(), code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, out=self.var, blocksize=self.blocksize, bits=self.bits)
        else:
            self.var = V.T.contiguous()

    def dequantize(self):
        if self.bits < 16:
            Vt = F.dequantize_blockwise(self.var, code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, outdtype=self.var_dtype, blocksize=self.blocksize, bits=self.bits)
            for j in range(self.rect_t1):
                Vt = 1.5 * Vt - 0.5 * Vt @ Vt.T @ Vt
        else:
            Vt = self.var

        return Vt.T @ self.Svalue.diag() @ Vt, Vt

    def computepower(self, exp, ridge_epsilon=1e-6):
        if self.bits < 16:
            Vt = F.dequantize_blockwise(self.var, code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, outdtype=self.var_dtype, blocksize=self.blocksize, bits=self.bits)
            return F.compute_power(Vt, self.Svalue, exp, iter_count=self.rect_t2, ridge_epsilon=ridge_epsilon)
        else:
            Vt = self.var
            return F.compute_power(Vt, self.Svalue, exp, iter_count=0, ridge_epsilon=ridge_epsilon)

    def set_device(self, device):
        if self.bits < 16:
            self.name2qmap[self.code] = self.name2qmap[self.code].to(device)
            self.absmax = self.absmax.to(device)
        self.Svalue = self.Svalue.to(device)
        self.var = self.var.to(device)
