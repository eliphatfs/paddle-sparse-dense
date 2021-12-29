import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paddle-sparse-dense", # Replace with your own username
    version="0.0.5",
    author="flandre.info",
    author_email="flandre@scarletx.cn",
    description="Sparse-dense operators for paddle.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eliphatfs/paddle-sparse-dense",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='~=3.6',
)
