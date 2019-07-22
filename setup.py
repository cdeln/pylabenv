# -*- coding: utf-8 -*-
import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name = 'pylabenv',
    version = '0.0.2',
    author = 'Carl Dehlin',
    author_email = 'carl@dehlin.com',
    description = 'A utility library for creating a Matlab-like environment for Python',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = '',
    license='MIT',
    classifiers = [
        # Add classifiers here
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    scripts = [ 
        'scripts/pylab'
    ],
    packages = setuptools.find_packages(),
    install_requires = [
        'numpy>=1.15.4',
        'scipy>=1.1.0',
        'opencv-python>=3.4.3.18',
        'matplotlib>=2.2.2',
        'pyyaml>=3.12',
        'pandas>=0.24.2'
    ]
)
