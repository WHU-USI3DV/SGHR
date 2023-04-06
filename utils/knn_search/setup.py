from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='knn_search',
    ext_modules=[
        CUDAExtension('knn_search', [
            './src/knn.cpp',
            './src/knn_cuda.cu',
        ], libraries=['cublas',])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)