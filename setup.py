import setuptools
from pathlib import Path

here = Path(__file__).absolute().parent

with open(here / 'README.md') as f:
    long_description = f.read()

setuptools.setup(
    name="fran",
    version='0.1.0',
    packages=["fran"],
    install_requires=[
        "imageio",
        "pygame",
        "pandas",
        "numpy",
        "scikit-image",
        "toml",
    ],
    package_data={"fran": ["config.toml"]},
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "fran = fran.__main__:main",
            "frame_annotator = fran.__main__:main",
        ]
    },
    url='https://github.com/clbarnes/frame_annotator',
    license='MIT',
    author='Chris L. Barnes',
    author_email='barnesc@janelia.hhmi.org',
    description='Annotate frames of a multiTIFF video'
)
