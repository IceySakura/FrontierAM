"""Compile the Garnet client in an isolated subprocess using the caller's Python."""

from __future__ import annotations

import sys

from pathlib import Path

import pybind11

from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


HIREDIS_SOURCES: tuple[str, ...] = ('alloc.c', 'async.c', 'hiredis.c', 'net.c', 'read.c', 'sds.c', 'sockcompat.c')


class GarnetBuildExt(build_ext):
    """Compile hiredis as C before linking it into the C++17 extension."""

    def build_extension(self, extension: Extension) -> None:
        hiredis_path = Path(extension.include_dirs[-1])
        extension.extra_objects = self.compiler.compile(
            [str(hiredis_path / name) for name in HIREDIS_SOURCES],
            output_dir=self.build_temp,
            include_dirs=[str(hiredis_path)],
            extra_postargs=['-O3', '-fvisibility=hidden'],
        )
        super().build_extension(extension)


def build_extension(directory: Path) -> None:
    """Write the extension and temporary objects beneath the supplied build directory."""
    native_path = Path(__file__).resolve().parent
    extension = Extension(
        '_garnet_native',
        sources=[str(native_path / 'garnet_client.cpp')],
        include_dirs=[pybind11.get_include(), str(native_path / 'vendor'), str(native_path / 'vendor/hiredis')],
        extra_compile_args=['-O3', '-std=c++17', '-fvisibility=hidden'],
        language='c++',
    )
    command = GarnetBuildExt(Distribution({'name': 'nexus-garnet-native', 'ext_modules': [extension]}))
    command.ensure_finalized()
    command.build_lib = str(directory)
    command.build_temp = str(directory / 'objects')
    command.force = True
    command.run()


if __name__ == '__main__':
    build_extension(Path(sys.argv[1]))
