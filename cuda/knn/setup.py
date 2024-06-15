from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='knn',
    ext_modules=[
        CUDAExtension('knn', [
            'knn_cuda.cpp',
            'knn.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
