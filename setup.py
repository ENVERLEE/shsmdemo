from setuptools import setup, find_packages

setup(
    name="research_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "plotly",
        "numpy",
        "scipy",
        "pandas",
        "requests"
    ]
)
