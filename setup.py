#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


requirements = (
'numpy', 'opencv-python', 'einops', 'solt==0.1.8', 'tqdm', 'scikit-learn', 'pandas',  'sas7bdat', 'pyyaml', 'matplotlib', 'coloredlogs==14.0', 'nibabel', 'torchio')

setup_requirements = ()

test_requirements = ('pytest',)

description = """CLIMATv2: Clinically-Inspired Multi-Agent Transformers for Disease Trajectory Forecasting from Multimodal Data
"""

setup(
    author="Huy Hoang Nguyen",
    author_email='huy.nguyen@oulu.fi',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux'
    ],
    description="CLIMATv2",
    install_requires=requirements,
    license="MIT license",
    long_description=description,
    include_package_data=True,
    keywords='',
    name='climatv2',
    packages=find_packages(include=[]),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Oulu-IMEDS/CLIMATv2',
    version='0.0.1',
    zip_safe=False,
)
