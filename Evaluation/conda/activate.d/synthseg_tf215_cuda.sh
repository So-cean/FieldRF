#!/usr/bin/env bash

if [[ -n "${LD_LIBRARY_PATH+x}" ]]; then
    export SYNTHSEG_TF215_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
    export SYNTHSEG_TF215_HAD_LD_LIBRARY_PATH=1
else
    unset SYNTHSEG_TF215_OLD_LD_LIBRARY_PATH
    export SYNTHSEG_TF215_HAD_LD_LIBRARY_PATH=0
fi

if [[ -n "${TF_FORCE_GPU_ALLOW_GROWTH+x}" ]]; then
    export SYNTHSEG_TF215_OLD_TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH}"
    export SYNTHSEG_TF215_HAD_TF_FORCE_GPU_ALLOW_GROWTH=1
else
    unset SYNTHSEG_TF215_OLD_TF_FORCE_GPU_ALLOW_GROWTH
    export SYNTHSEG_TF215_HAD_TF_FORCE_GPU_ALLOW_GROWTH=0
fi

_synthseg_tf215_site="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
_synthseg_tf215_cuda_libs="${_synthseg_tf215_site}/cublas/lib:${_synthseg_tf215_site}/cuda_cupti/lib:${_synthseg_tf215_site}/cuda_nvcc/lib:${_synthseg_tf215_site}/cuda_nvrtc/lib:${_synthseg_tf215_site}/cuda_runtime/lib:${_synthseg_tf215_site}/cudnn/lib:${_synthseg_tf215_site}/cufft/lib:${_synthseg_tf215_site}/curand/lib:${_synthseg_tf215_site}/cusolver/lib:${_synthseg_tf215_site}/cusparse/lib:${_synthseg_tf215_site}/nccl/lib:${_synthseg_tf215_site}/nvjitlink/lib"
export LD_LIBRARY_PATH="${_synthseg_tf215_cuda_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
unset _synthseg_tf215_site _synthseg_tf215_cuda_libs
