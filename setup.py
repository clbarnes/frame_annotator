import setuptools
from pathlib import Path

here = Path(__file__).absolute().parent

with open(here / 'README.md') as f:
    long_description = f.read()

name = "frame_annotator"
requirements = [
    "imageio",
    "pygame",
    "pandas",
    "numpy",
    "scikit-image",
    "toml",
]

setuptools.setup(
    name=name,
    version='0.1.0',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    package_data={"frame_annotator": ["config.toml"]},
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7",
    entry_points={"console_scripts": ["frame_annotator = frame_annotator:main"]},
    url='https://github.com/clbarnes/frame_annotator',
    license='MIT',
    author='Chris L. Barnes',
    author_email='',
    description=''
)
