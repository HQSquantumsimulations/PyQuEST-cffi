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
import sys
import os
#import git


class CustomExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CustomBuild(build_ext):
    def run(self):
        try:
            subprocess.run(['make', '--version'], check=True)
        except OSError:
            raise RuntimeError(
                "Make must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if not ext.name == 'questlib':
            pass
        else:
            old_path = os.getcwd()
            # os.path.dirname(os.path.abspath(sys.argv[0]))
            src_path = os.path.dirname(os.path.realpath(__file__))
            os.chdir(src_path)
            QuEST_release_link = 'https://github.com/HQSquantumsimulations/QuEST/archive/develop_damping_v0.1.tar.gz'
            # try:
            #repo = git.Repo()
            #repo.submodule_update(init=True, recursive=True)
            #    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'], check=True)
            # except Exception:
            if not os.path.exists((os.path.realpath(__file__)+'/QuEST')):
                os.makedirs(os.path.realpath(__file__)+'/QuEST')
                subprocess.run(['wget',
                                QuEST_release_link,
                                '-o',
                                os.path.realpath(__file__)+'/QuEST.tar.gz'],
                               check=True)
                subprocess.run(['tar',
                                '-xzvf',
                                os.path.realpath(__file__)+'/QuEST.tar.gz',
                                '-C',
                                os.path.realpath(__file__)+'/QuEST/',
                                '--strip-components=1'],
                               check=True)
                # else:
                #    raise RuntimeError(
                #        'Could not update QuEST submodule but QuEST folder exists so can not download')

            os.chdir(src_path+'/pyquest_cffi/questlib/')
            subprocess.run(['make'], check=True)
            subprocess.run(['python', 'build_quest.py'], check=True)
            os.chdir(old_path)


class BuildPyCommand(build_py.build_py):
    """Custom build command."""

    def run(self):
        self.run_command('build_ext')
        build_py.build_py.run(self)


def setup_packages():

    with open('README.md') as file:
        readme = file.read()

    with open('LICENSE') as file:
        license = file.read()

    install_requires = [
        'cffi',
        'numpy',
        'pytest',
        'gitpython',
    ]
    packages = find_packages(exclude=('docs'))

    setup_args = {'name': 'pyquest_cffi',
                  'description': ('Provides: Interactive python interface to QuEST quantum simulation toolkit;'
                                  + '  Compile functionality, create, build and import valid QuEST source code from python'),
                  'version': '0.0.1',
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
                  'package_data': {'pyquest_cffi': ['test', 'questlib/*', 'questlib/*.so']},
                  'ext_modules': [CustomExtension('questlib')],
                  # add custom build_ext command
                  'cmdclass': {'build_ext': CustomBuild,
                               'build_py': BuildPyCommand},
                  'zip_safe': False,
                  }
    setup(**setup_args)


setup_packages()
