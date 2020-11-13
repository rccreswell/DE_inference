from setuptools import setup

setup(
    name='diffeqinf',
    description='Inference for time series with numerical error',
    version='0.0.1a',
    install_requires=[
        'matplotlib>=3.2',
        'numpy>=1.17',
        'pints',
        'pytest',
        'scipy>=1.3'
    ]
)
