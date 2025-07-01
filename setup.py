from setuptools import find_packages, setup

with open("README.md", "r", encoding='utf-8') as f:
    long_description = f.read()

# 开发依赖 - 用于测试、格式化、类型检查等
dev_requirements = [
    'pytest>=6.0.0',
    'pytest-cov>=2.10.0',
    'pytest-xdist>=2.0.0',  # 并行测试
    'yapf==0.32.0',  # 代码格式化
    'isort==5.10.1',  # 导入排序
    'flake8>=3.8.0',  # 代码检查
    'mypy>=0.800',  # 类型检查
    'bandit>=1.7.0',  # 安全检查
    'coverage>=5.0.0',  # 覆盖率
    'toml>=0.10.2',  # TOML解析，yapf需要
]

setup(
    name='torch-rechub',
    version='0.1.0',
    description='A Pytorch Toolbox for Recommendation Models, Easy-to-use and Easy-to-extend.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Datawhale',
    author_email='laimc@shanghaitech.edu.cn',
    url='https://github.com/datawhalechina/torch-rechub',
    install_requires=['numpy>=1.19.0',
                      'torch>=1.7.0',
                      'pandas>=1.0.5',
                      'tqdm>=4.64.0',
                      'scikit_learn>=0.23.2'],
    extras_require={
        'dev': dev_requirements,
        'test': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'pytest-xdist>=2.0.0',
        ],
    },
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
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
    ],
    keywords=['ctr',
              'click through rate',
              'deep learning',
              'pytorch',
              'recsys',
              'recommendation'],
    python_requires='>=3.8',
)
