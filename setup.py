# The wheel is the C++ binary (make, metallib embedded) plus py/mmmllm's thin CLI; any Python 3, macOS 15+ arm64.
import shutil
import subprocess

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        subprocess.run(["make", "-j", "build/mllm"], check=True)
        shutil.copy2("build/mllm", f"{self.build_lib}/mmmllm/mllm")


class Wheel(bdist_wheel):
    def get_tag(self):
        return "py3", "none", "macosx_15_0_arm64"  # Makefile's MACOSX_DEPLOYMENT_TARGET


setup(cmdclass={"build_py": BuildPy, "bdist_wheel": Wheel})
