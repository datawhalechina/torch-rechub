from distutils.core import setup
from setuptools import find_packages

with open("README.md", "r", encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='torch-rechub',
    version='0.0.2',
    description='A Lighting Pytorch Framework for Recommendation System, Easy-to-use and Easy-to-extend.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Datawhale',
    author_email='laimc@shanghaitech.edu.cn',
    url='https://github.com/datawhalechina/torch-rechub',
    install_requires=['numpy>=1.19.0', 'torch>=1.7.0', 'pandas>=1.0.5', 'tqdm>=4.64.0', 'scikit_learn>=0.23.2', 'annoy>=1.17.0'],
    packages=find_packages(),
    platforms=["all"],
    classifiers=[
        'Intended Audience :: Developers',
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
    ],
    keywords=['ctr', 'click through rate', 'deep learning', 'pytorch', 'recsys', 'recommendation'],
)
