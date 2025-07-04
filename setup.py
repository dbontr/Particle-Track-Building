# setup.py
from setuptools import setup, find_packages

setup(
  name="trackml_reco",
  version="0.1",
  packages=find_packages(),   # will find trackml_reco
  install_requires=[
    "numpy",
    "pandas",
    "scipy",
    "networkx",
    "matplotlib",
    # “trackml” comes from GitHub via requirements.txt
  ],
  entry_points={
    "console_scripts": [
      "trackml-reco = trackml_reco.main:main",
    ]
  }
)
