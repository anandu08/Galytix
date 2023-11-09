from setuptools import setup, find_packages

setup(
    name='Galytix',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'gensim',
        'numpy',
        'chardet',
    ],
)
