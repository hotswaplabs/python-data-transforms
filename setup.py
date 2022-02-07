from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Python package'
LONG_DESCRIPTION = 'Collection of data transformations for Pandas'

# Setting up
setup(
        name="transforms", 
        version=VERSION,
        author="Hotswap",
        author_email="<team@hotswap.app>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['pandas', 'plotly', 'numpy'],
        keywords=[],
        classifiers=[]
)
