import setuptools

RELEASE_VERSION = 'v0.2.0'

with open('requirements.txt', 'r') as fh:
    requirements = fh.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autocontrastive-gen",
    version=f"{RELEASE_VERSION}".replace('v', ''),
    author="IBM Research",
    author_email="ariel.gera1@ibm.com",
    url="https://github.com/IBM/auto-contrastive-generation",
    description="Auto-Contrastive Text Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    license='Apache License 2.0',
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    package_data={"": ["LICENSE", "requirements.txt"]},
    include_package_data=True
)
