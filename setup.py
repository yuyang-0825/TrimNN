from setuptools import setup, find_packages

setup(
    name="TrimNN",
    version="0.0.1",
    author="Yang Yu",
    author_email="yykk3@umsystem.edu",
    description="TrimNN: an empowered bottom-up approach designed to estimate the prevalence of sizeable CC motifs in a triangulated cell graph",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yuyang-0825/TrimNN",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.9',
    install_requires=[
        'tqdm>=4.66.1',
        'numpy>=1.25.2',
        'pandas>=2.0.3',
        'scipy>=1.11.2',
        'scikit-learn>=1.3.0',
        'python-igraph>=0.9.6',
        'tensorboard>=2.6.2.2',
        'networkx>=3.1',
        'torch>=1.13.1',
        'dgl>=1.1.2',
        'matplotlib>=3.7.2',

    ],
)
