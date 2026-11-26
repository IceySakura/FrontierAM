"""Small x86-64 AVX2/FMA convolution code generator.

The generated functions implement only dense float32 convolution, optional
ReLU, and a return. They use the standard System V ABI and no runtime services.
"""
import ctypes
import mmap
import struct

import numpy as np


class Assembler:
    def __init__(self):
        self.code = bytearray()
        self.labels = {}
        self.fixups = []

    def emit(self, *values):
        self.code.extend(values)

    def imm32(self, value):
        self.code.extend(struct.pack("<i", value))

    def label(self, name):
        self.labels[name] = len(self.code)

    def jump_not_zero(self, name):
        self.emit(0x0f, 0x85)
        self.fixups.append((len(self.code), name))
        self.imm32(0)

    def vex(self, opcode, dst, src1=0, src2=None, base=None, disp=0,
            opcode_map=1, prefix=0):
        b = (src2 >> 3) if src2 is not None else (base >> 3)
        self.emit(0xc4,
                  ((1-(dst >> 3)) << 7) | (1 << 6) | ((1-b) << 5) | opcode_map,
                  ((15-src1) << 3) | (1 << 2) | prefix,
                  opcode)
        if src2 is not None:
            self.emit(0xc0 | ((dst & 7) << 3) | (src2 & 7))
        else:
            self.emit(0x80 | ((dst & 7) << 3) | (base & 7))
            self.imm32(disp)

    def finish(self):
        for offset, name in self.fixups:
            struct.pack_into("<i", self.code, offset, self.labels[name]-offset-4)
        return bytes(self.code)


def generate(cin, cout, kernel, width, relu):
    if cout not in (8, 16, 24, 32):
        raise ValueError("native convolution supports 8, 16, 24 or 32 output channels")
    vectors = cout//8
    pixels = 4 if cout == 24 else min(4, 8//vectors)
    if width % pixels:
        raise ValueError("width must be a multiple of the native pixel group")
    a = Assembler()
    # rdi: padded input, rsi: weights, rdx: bias, rcx: output, r8d: height.
    if relu and cout != 24:
        a.vex(0x57, 14, 14, 14)
    a.label("row")
    a.emit(0x41, 0xb9)
    a.imm32(width//pixels)
    a.label("pixel")
    for p in range(pixels):
        for v in range(vectors):
            a.vex(0x10, p*vectors+v, base=2, disp=v*32)
    for ky in range(kernel):
        for kx in range(kernel):
            for c in range(cin):
                feature = (ky*kernel+kx)*cin+c
                offset = ((ky*(width+kernel-1)+kx)*cin+c)*4
                for v in range(vectors):
                    weight_reg = 13+v if cout == 24 else 8+pixels+v
                    a.vex(0x10, weight_reg, base=6, disp=(feature*cout+v*8)*4)
                for p in range(pixels):
                    input_reg = 12 if cout == 24 else 8+p
                    a.vex(0x18, input_reg, base=7, disp=offset+p*cin*4,
                          opcode_map=2, prefix=1)
                    for v in range(vectors):
                        weight_reg = 13+v if cout == 24 else 8+pixels+v
                        a.vex(0xb8, p*vectors+v, input_reg, weight_reg,
                              opcode_map=2, prefix=1)
    zero_reg = 12 if cout == 24 else 14
    if relu and cout == 24:
        a.vex(0x57, zero_reg, zero_reg, zero_reg)
    for p in range(pixels):
        for v in range(vectors):
            reg = p*vectors+v
            if relu:
                a.vex(0x5f, reg, reg, zero_reg)
            a.vex(0x11, reg, base=1, disp=(p*cout+v*8)*4)
    a.emit(0x48, 0x81, 0xc7)
    a.imm32(pixels*cin*4)
    a.emit(0x48, 0x81, 0xc1)
    a.imm32(pixels*cout*4)
    a.emit(0x41, 0xff, 0xc9)
    a.jump_not_zero("pixel")
    a.emit(0x48, 0x81, 0xc7)
    a.imm32((kernel-1)*cin*4)
    a.emit(0x41, 0xff, 0xc8)
    a.jump_not_zero("row")
    a.emit(0xc5, 0xf8, 0x77, 0xc3)
    return a.finish()


class Convolution:
    def __init__(self, cin, cout, kernel, width=512, relu=True, code=None):
        self.cin, self.cout = cin, cout
        self.kernel, self.width = kernel, width
        if code is None:
            code = generate(cin, cout, kernel, width, relu)
        self.memory = mmap.mmap(-1, len(code),
                                prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        self.memory.write(code)
        address = ctypes.addressof(ctypes.c_char.from_buffer(self.memory))
        signature = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
        self.function = signature(address)

    def __call__(self, a, weights, bias):
        h, w, cin = a.shape
        if w != self.width or cin != self.cin or h < 1:
            raise ValueError("convolution shape does not match compiled kernel")
        r = self.kernel//2
        if r:
            a = np.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
        else:
            a = np.ascontiguousarray(a)
        out = np.empty((h, w, self.cout), np.float32)
        self.function(a.ctypes.data, weights.ctypes.data, bias.ctypes.data,
                      out.ctypes.data, h)
        return out


def available():
    from numpy.core._multiarray_umath import __cpu_features__
    return bool(__cpu_features__.get("AVX2") and __cpu_features__.get("FMA3"))


def main():
    import time
    from solution import conv, limit_blas_threads
    limit_blas_threads()
    rng = np.random.default_rng(1)
    print("available", available(), flush=True)
    for cin, cout, kernel in [(4, 16, 5), (16, 16, 3), (16, 16, 1),
                               (4, 32, 5), (32, 32, 3), (32, 16, 1),
                               (1, 24, 9), (24, 24, 3), (24, 8, 1)]:
        a = rng.normal(size=(20, 512, cin)).astype(np.float32)
        w = rng.normal(size=(kernel*kernel*cin, cout)).astype(np.float32)*0.1
        b = rng.normal(size=cout).astype(np.float32)*0.1
        for relu in [False, True]:
            function = Convolution(cin, cout, kernel, relu=relu)
            t = time.perf_counter()
            result = function(a, w, b)
            seconds = time.perf_counter()-t
            expected = conv(a, w, b, kernel, relu=relu)
            difference = np.abs(result-expected)
            print(cin, cout, kernel, relu, "max_diff", difference.max(),
                  "rmse", np.sqrt(np.mean(difference**2)), "time", seconds, flush=True)
            assert difference.max() < 1e-4


if __name__ == "__main__":
    main()
