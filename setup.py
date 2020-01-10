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
import os


class CustomExtension(Extension):
    """Custom Extension class"""

    def __init__(self, name, sourcedir=''):
        """Initialise custom extension"""
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CustomBuild(build_ext):
    """Custom C builder"""

    def run(self):
        """Run custom build function"""
        try:
            subprocess.run(['make', '--version'], check=True)
        except OSError:
            raise RuntimeError(
                "Make must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        """Build extension"""
        if not ext.name == 'questlib':
            pass
        else:
            old_path = os.getcwd()
            # os.path.dirname(os.path.abspath(sys.argv[0]))
            src_path = os.path.dirname(os.path.realpath(__file__))
            os.chdir(src_path)
            QuEST_release_link = 'https://github.com/QuEST-Kit/QuEST/archive/3.0.0.tar.gz'

            if not os.path.exists((src_path + '/QuEST')):
                os.makedirs(src_path + '/QuEST/')
                subprocess.run(['wget',
                                QuEST_release_link,
                                '-O',
                                src_path + '/QuEST.tar.gz'],
                               check=True)
                subprocess.run(['tar',
                                '-xzvf',
                                src_path + '/QuEST.tar.gz',
                                '-C',
                                src_path + '/QuEST/',
                                '--strip-components=1'],
                               check=True)
            os.chdir(src_path + '/pyquest_cffi/questlib/')
            subprocess.run(['make'], check=True)
            subprocess.run(['python', 'build_quest.py'], check=True)
            os.chdir(old_path)


class BuildPyCommand(build_py.build_py):
    """Custom build command."""

    def run(self):
        """Run python build"""
        self.run_command('build_ext')
        build_py.build_py.run(self)


def setup_packages():
    """Setup method"""
    with open('README.md') as file:
        readme = file.read()

    with open('LICENSE') as file:
        license = file.read()

    install_requires = [
        'cffi',
        'numpy',
        'pytest',
    ]
    packages = find_packages(exclude=('docs'))

    setup_args = {'name': 'pyquest_cffi',
                  'description': (
                      'Provides: Interactive python interface'
                      + ' to QuEST quantum simulation toolkit;'
                      + '  Compile functionality, create, build and import'
                      + ' valid QuEST source code from python'),
                  'version': '3.0.0',
                  'long_description': readme,
                  'packages': packages,
                  # 'package_dir': {'': 'pyquest_cffi'},
                  'author': 'HQS Quantum Simulations: Sebastian Zanker, Nicolas Vogt',
                  'author_email': 'info@quantumsimulations.de',
                  'url': '',
                  'download_url': '',
                  'license': license,
                  'install_requires': install_requires,
                  'setup_requires': ['cffi'],
                  'include_package_data': True,
                  'package_data': {'pyquest_cffi': ['questlib/*', 'questlib/*.so']},
                  'data_files': [("", ["LICENSE", "pyquest_cffi/questlib/makefile"])],
                  'ext_modules': [CustomExtension('questlib')],
                  # add custom build_ext command
                  'cmdclass': {'build_ext': CustomBuild,
                               'build_py': BuildPyCommand},
                  'zip_safe': False,
                  }
    setup(**setup_args)


setup_packages()
