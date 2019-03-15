from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED = ["matplotlib", "torch", "tqdm", "numpy", "opencv"]
DEPENDENCY_LINKS = ["git+ssh://github.com/hassony2/chumpy.git"]

setup(
    name="manopth-hassony2",
    version="0.0.1",
    author="Yana Hasson",
    author_email="yana.hasson.inria@gmail.com",
    packages=find_packages(exclude=('tests',)),
    python_requires=">=3.5.0",
    description="PyTorch mano layer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=REQUIRED,
    dependency_links=DEPENDENCY_LINKS,
    url="https://github.com/hassony2/manopth",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
)
