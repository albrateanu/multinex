#!/usr/bin/env python

from setuptools import find_packages, setup
import os
import subprocess
import sys
import time
from pathlib import Path

# -------------------------
# Version helpers
# -------------------------
version_file = 'basicsr/version.py'

def readme():
    return '' 

def _minimal_ext_cmd(cmd):
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out

def get_git_hash():
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'
    return sha

def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from basicsr.version import __version__
            sha = __version__.split('+')[-1]
        except Exception:
            sha = 'unknown'
    else:
        sha = 'unknown'
    return sha

def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
short_version = '{}'
version_info = ({})
"""
    sha = get_hash()
    with open('VERSION', 'r') as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = ', '.join(
        [x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split('.')])
    VERSION = SHORT_VERSION + '+' + sha

    version_file_str = content.format(time.asctime(), VERSION, SHORT_VERSION, VERSION_INFO)
    Path(version_file).parent.mkdir(parents=True, exist_ok=True)
    with open(version_file, 'w') as f:
        f.write(version_file_str)

def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

# -------------------------
# Extension helpers
# -------------------------
def want_cuda_ext():
    """
    Decide whether to even attempt building native extensions.
    - Command-line: --no_cuda_ext disables
    - Env: BASICSR_EXT in {1,true,yes} enables; {0,false,no} disables
    Default: ENABLE (to mirror upstream), but you can set BASICSR_EXT=False to skip.
    """
    if '--no_cuda_ext' in sys.argv:
        sys.argv.remove('--no_cuda_ext')
        return False

    val = os.getenv('BASICSR_EXT', None)
    if val is None:
        return True
    return val.strip().lower() in {'1', 'true', 'yes', 'on'}

def make_cuda_or_cpp_ext(name, module, sources, sources_cuda=None):
    """
    Creates either a CUDAExtension (if CUDA available) or a CppExtension fallback.
    Import torch/cpp_extension lazily and only if we actually build extensions.
    """
    sources_cuda = sources_cuda or []

    try:
        import torch
        from torch.utils.cpp_extension import (BuildExtension, CppExtension, CUDAExtension)
    except Exception:
        return None, None, None

    define_macros = []
    extra_compile_args = {'cxx': []}

    use_cuda = False
    try:
        use_cuda = torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1'
    except Exception:
        use_cuda = False

    if use_cuda:
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        all_sources = sources + sources_cuda
    else:
        # No CUDA at build time -> pure C++ fallback
        extension = CppExtension
        all_sources = sources

    ext = extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in all_sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args
    )
    return ext, BuildExtension, extension

# -------------------------
# Main setup
# -------------------------
if __name__ == '__main__':
    write_version_py()

    ext_modules = []
    cmdclass = {}

    if want_cuda_ext():
        entries = [
            dict(
                name='deform_conv_ext',
                module='basicsr.models.ops.dcn',
                sources=['src/deform_conv_ext.cpp'],
                sources_cuda=['src/deform_conv_cuda.cpp', 'src/deform_conv_cuda_kernel.cu'],
            ),
            dict(
                name='fused_act_ext',
                module='basicsr.models.ops.fused_act',
                sources=['src/fused_bias_act.cpp'],
                sources_cuda=['src/fused_bias_act_kernel.cu'],
            ),
            dict(
                name='upfirdn2d_ext',
                module='basicsr.models.ops.upfirdn2d',
                sources=['src/upfirdn2d.cpp'],
                sources_cuda=['src/upfirdn2d_kernel.cu'],
            ),
        ]

        build_ext_cls = None
        for e in entries:
            ext, BuildExtension, _ = make_cuda_or_cpp_ext(**e)
            if ext is None:
                ext_modules = []
                build_ext_cls = None
                break
            ext_modules.append(ext)
            build_ext_cls = BuildExtension

        if build_ext_cls is not None:
            cmdclass['build_ext'] = build_ext_cls

    setup(
        name='basicsr',
        version=get_version(),
        description='Open Source Image and Video Super-Resolution Toolbox',
        long_description=readme(),
        author='Xintao Wang',
        author_email='xintao.wang@outlook.com',
        keywords='computer vision, restoration, super resolution',
        url='https://github.com/xinntao/BasicSR',
        packages=find_packages(exclude=('options', 'datasets', 'experiments', 'results', 'tb_logger', 'wandb')),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        install_requires=[],
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        zip_safe=False
    )

