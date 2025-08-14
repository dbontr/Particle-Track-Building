from setuptools import setup, find_packages

setup(
    name="trackml_reco",
    version="0.1.0",
    description="EKF-based track-building framework with multiple branchers for TrackML",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # GitHub dependency for TrackML library
        "trackml @ git+https://github.com/LAL/trackml-library.git@master",
        
        # Runtime dependencies
        "numpy",
        "numba",
        "pandas",
        "matplotlib",
        "scipy",
        "networkx",
        "orjson",
    ],
    extras_require={
        # Optional speed/profiling stack
        "speed": [
            "scalene>=1.5.49; platform_system != 'Windows'",
            "py-spy>=0.3.14",
        ],
        # Developer extras
        "dev": [
            "pytest",
            "black",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            # CLI entry point for running main.py
            "trackml-reco=trackml_reco.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
