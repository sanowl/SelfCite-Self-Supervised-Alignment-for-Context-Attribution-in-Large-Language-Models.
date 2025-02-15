from setuptools import setup, find_packages

setup(
    name="SelfCite",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "nltk",
    ],
    entry_points={
        "console_scripts": [
            "selfcite = main:main",
        ],
    },
)
