"""Setup for PyQuest-cffi"""
# Copyright 2019 HQS Quantum Simulations GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command import build_py
import subprocess
import platform
import os
path = os.path.dirname(os.path.abspath(__file__))


class CustomExtension(Extension):
    """Custom Extension class"""

    def __init__(self, name, sourcedir=''):
        """Initialise custom extension"""
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CustomBuild(build_ext):
    """Custom C builder"""

    def run(self) -> None:
        """Run custom build function"""
        # try:
        #     subprocess.run(['make', '--version'], check=True)
        # except OSError:
        #     raise RuntimeError(
        #         "Make must be installed to build pyquest"
        #     )
        try:
            subprocess.run(['wget', '--version'], check=True)
        except OSError:
            raise RuntimeError(
                "Wget must be installed to build pyquest "
            )
        try:
            subprocess.run(['tar', '--version'], check=True)
        except OSError:
            raise RuntimeError(
                "Tar must be installed to build pyquest: "
            )
        try:
            subprocess.run(['cmake', '--version'], check=True)
        except OSError:
            raise RuntimeError(
                "Cmake must be installed to build pyquest"
            )
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: Extension) -> None:
        """Build extension"""
        if not ext.name == 'questlib':
            pass
        else:
            old_path = os.getcwd()
            # os.path.dirname(os.path.abspath(sys.argv[0]))
            print('__file__', __file__)
            src_path = os.path.dirname(os.path.realpath(__file__))
            print('src_path', src_path)
            os.chdir(src_path)
            QuEST_release_link = 'https://github.com/QuEST-Kit/QuEST/archive/v3.2.0.tar.gz'
            # https://github.com/QuEST-Kit/QuEST/releases/tag/v3.2.0
            if not os.path.exists(os.path.join(src_path, 'QuEST')):
                print('quest_path', os.path.join(src_path, 'QuEST/'))
                os.makedirs(os.path.join(src_path, 'QuEST/'))
                subprocess.run(['wget',
                                QuEST_release_link,
                                '-O',
                                os.path.join(src_path, 'QuEST.tar.gz')],
                               check=True)
                subprocess.run(['tar',
                                '-xzvf',
                                os.path.join(src_path, 'QuEST.tar.gz'),
                                '-C',
                                os.path.join(src_path, 'QuEST/'),
                                '--strip-components=1'],
                               check=True)
            # Switch off Multithreading on macos because of openmp problems
            if platform.system() == 'Darwin':
                # Disabling implicit function declaration warning in C compilation
                if 'CFLAGS' in os.environ.keys():
                    os.environ['CFLAGS'] = os.environ['CFLAGS'] + " -Wno-implicit-function-declaration"
                else:
                    os.environ['CFLAGS'] = "-Wno-implicit-function-declaration"
                args_for_cmake = ['-DMULTITHREADED=0', "-DCMAKE_POLICY_DEFAULT_CMP0012=NEW"]
            else:
                args_for_cmake = ["-DCMAKE_POLICY_DEFAULT_CMP0012=NEW"]

            os.chdir(os.path.join(src_path, 'pyquest_cffi/questlib/'))
            run_command_make = ['cmake', os.path.join(src_path, 'QuEST/QuEST/')] + args_for_cmake
            run_command_build = ['cmake', '--build', '.']
            subprocess.run(run_command_make, check=True)
            subprocess.run(run_command_build, check=True)
            from build_quest import build_quest_so
            build_quest_so()
            print(os.listdir('.'))
            os.chdir(old_path)


class BuildPyCommand(build_py.build_py):
    """Custom build command."""

    def run(self):
        """Run python build"""
        self.run_command('build_ext')
        build_py.build_py.run(self)


def setup_packages():
    """Setup method"""
    with open(os.path.join(path, 'README.md')) as file:
        readme = file.read()

    install_requires = [
        'cffi',
        'numpy',
    ]
    packages = find_packages(exclude=('docs'))

    setup_args = {'name': 'pyquest_cffi',
                  'description': (
                      'Provides: Interactive python interface'
                      + ' to QuEST quantum simulation toolkit;'
                      + '  Compile functionality, create, build and import'
                      + ' valid QuEST source code from python'),
                  'version': '3.2.4',
                  'long_description': readme,
                  'long_description_content_type': 'text/markdown',
                  'packages': packages,
                  # 'package_dir': {'': 'pyquest_cffi'},
                  'author': 'HQS Quantum Simulations: Sebastian Zanker, Nicolas Vogt',
                  'author_email': 'info@quantumsimulations.de',
                  'url': '',
                  'download_url': 'https://github.com/HQSquantumsimulations/PyQuEST-cffi/archive/3.2.3.tar.gz',
                  'license': 'Apache-2.0',
                  'install_requires': install_requires,
                  'setup_requires': ['cffi'],
                  # 'include_package_data': True,
                  'package_data': {'pyquest_cffi': ['questlib/*', 'questlib/*.so',  'questlib/*.dylib']},
                  'data_files': [("", ["LICENSE"])],
                  # add custom build_ext command
                  'cmdclass': {'build_ext': CustomBuild,
                               'build_py': BuildPyCommand},
                  'zip_safe': False,
                  'python_requires': '>=3.6',
                  'ext_modules': [CustomExtension('questlib')],
                  }
    setup(**setup_args)


setup_packages()
