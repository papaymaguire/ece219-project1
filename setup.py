from setuptools import find_packages, setup

setup(
    name="project1",
    packages=find_packages(exclude=["project1_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "nltk",
        "jupyter",
        "notebook",
        "dagstermill",
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
