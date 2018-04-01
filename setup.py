import sys
import os
from setuptools import setup, find_packages

# read version file
with open(os.path.join('neighborhood', 'VERSION'), 'r') as fp:
    version = fp.read().strip('\n')

# block unintentional install in Python 2.x
if sys.version_info.major < 3:
    sys.exit("MACE requires Python 3")

# read README
with open('README.rst', 'r') as fp:
    readme=fp.read()

# install
setup(
    name='neighborhood',
    version=version,
    description='Neighborhood Algorithm Optimization and Ensemble Appraisal',
    long_description=readme,
    author='Keith Ma',
    author_email='keithfma@gmail.com',
    url='https://github.com/keithfma/neighborhood',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'pytest',
        ],
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        ]
)
