from setuptools import setup
from pathlib import Path

here = Path(__file__).absolute().parent

with open(here / 'README.md') as f:
    long_description = f.read()

setup(
    name='frame_annotator',
    version='0.1.0',
    packages=['frame_annotator'],
    install_requires=[
        "imageio",
        "pygame",
        "async_lru",
        "pandas",
        "numpy",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7",
    entry_points={"console_scripts": ["frame_annotator = frame_annotator.main"]},
    url='https://github.com/clbarnes/frame_annotator',
    license='MIT',
    author='Chris L. Barnes',
    author_email='',
    description=''
)
