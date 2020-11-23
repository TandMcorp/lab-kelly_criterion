import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kelly_criterion",
    version="0.0.1",
    author="Kelly Criterion",
    description="Exploring basic properties of the Kelly Criterion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
    ]
)
