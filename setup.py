import setuptools

with open("README.md") as fh:
    long_description = fh.read()

setuptools.setup(
    name="unsupervised_behaviors",
    version="0.0.1",
    author="Benjamin Wild",
    author_email="b.w@fu-berlin.de",
    url="https://github.com/nebw/unsupervised_behaviors/",
    description="Unsupervised ethograms using world models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.6",
    install_requires=[
        "h5py",
        "hdf5plugin",
        "numpy",
        "pandas",
        "scikit-image",
        "scikit-learn",
        "torch",
        "torchvision",
        "torchtyping",
        "pytest",
    ],
    extras_require={
        "beesbook_data": [
            "bb_behavior @ git+ssh://git@github.com/BioroboticsLab/bb_behavior.git#egg=master",
            "bb_tracking @ git+ssh://git@github.com/walachey/bb_tracking.git#egg=master",
            "bb_pipeline @ git+ssh://git@github.com/BioroboticsLab/bb_pipeline.git#egg=master",
        ],
        "vae_model": [
            "hierarchical_vae @ git+ssh://git@github.com/nebw/hierarchical_vae.git#egg=master",
        ],
    },
)
