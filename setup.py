import setuptools
from pathlib import Path


workdir = Path(__file__).parent.absolute() / 'axDo/workdirs/dev'
workdir.mkdir(parents=True, exist_ok=True)
workdir = Path(__file__).parent.absolute() / 'axDo/results'
workdir.mkdir(parents=True, exist_ok=True)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EEG-Cluster", # Replace with your own username
    version="0.1",
    author="xavante",
    author_email="xavante.erickson@gmail.com",
    description="mvpa_package for Lund university",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points = {
        'console_scripts': ['eray=SetupCluster:main'],
    },
    packages=setuptools.find_packages(exclude=["data", "documents", "Old", "Orchestration", "Plotting", "ReadableMatlab"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.8.5',
    install_requires=[
        'ray==1.2.0',
        'scipy==1.6.1',
        'scikit-learn==0.24.1',
        'scikit-image==0.18.1',
        'openstacksdk>=0.54.0',
        'accelerator',
    ]
)