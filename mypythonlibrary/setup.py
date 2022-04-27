from setuptools import find_packages, setup

setup(
    name = 'mypythonlib',
    packages = find_packages(include=['mypythonlib']),
    version = '0.1.0',
    description = 'Python library Scientific Computing',
    author = 'Thea Barnes',
    license = 'MIT',
)
