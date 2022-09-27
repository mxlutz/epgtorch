from setuptools import setup, find_packages

setup(
    name='epgtorch',
    version='0.1',
    url='https://github.com/fzimmermann89/epgtorch.git',
    author='Felix Zimmermann',
    author_email='felix.zimmermann@ptb.de',
    description='simple epg for pytorch',
    packages=find_packages(),    
    install_requires=['torch >= 1.10'],
)