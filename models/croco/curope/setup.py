# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# builds curope for rtx 8000 (sm_75), a100/a800 (sm_80), h100 (sm_90) with a ptx fallback
# respects TORCH_CUDA_ARCH_LIST (e.g., "7.5;8.0;9.0+PTX" or "8.0;9.0a+PTX")

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def parse_torch_arch_list(env_value: str):
    # accepts tokens like "7.5", "9.0", "9.0a", "9.0+PTX"
    arch_list, ptx_cc = [], None
    for token in [t.strip() for t in env_value.split(";") if t.strip()]:
        if token.endswith("+PTX"):
            base = token[:-4]                      # e.g., "9.0"
            base_clean = base.replace(".", "")     # "90"
            arch_list.append(base_clean)           # build sass for sm_90
            ptx_cc = base_clean.replace("a", "")   # ptx target can't be 90a -> use 90
        else:
            arch_list.append(token.replace(".", ""))  # keep possible '90a'
    return arch_list, ptx_cc


def make_gencode_flags(arch_list, ptx_cc=None):
    # emits -gencode entries for each arch; optional ptx fallback for forward compat
    flags = []
    for cc in arch_list:
        sm = f"sm_{cc}"                         # supports e.g. sm_90a
        compute = cc.replace("a", "")           # compute_90 for 90/90a
        flags += ["-gencode", f"arch=compute_{compute},code={sm}"]
    if ptx_cc:
        flags += ["-gencode", f"arch=compute_{ptx_cc},code=compute_{ptx_cc}"]
    return flags


# decide targets: env override or sensible defaults for your cluster
arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
if arch_env:
    target_archs, ptx_target = parse_torch_arch_list(arch_env)
else:
    target_archs, ptx_target = ["75", "80", "90"], "90"  # rtx8000 + a100 + h100

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--ptxas-options=-O3,-v",
] + make_gencode_flags(target_archs, ptx_cc=ptx_target)

cxx_flags = ["-O3", "-std=c++17", "-DNDEBUG"]

setup(
    name="curope",
    ext_modules=[
        CUDAExtension(
            name="curope",
            sources=[
                "curope.cpp",
                "kernels.cu",
            ],
            extra_compile_args={"nvcc": nvcc_flags, "cxx": cxx_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)

# # see what torch you actually run with
# python -c "import torch,sys,os; print('torch=',torch.__version__,' lib=',os.path.dirname(torch.__file__))"

# # remove the bad build
# pip uninstall -y curope

# # build against the torch in *this* venv (no temp overlay)
# export TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0+PTX"
# pip install -v --no-build-isolation -e models/croco/curope