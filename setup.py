from setuptools import setup, find_packages
  
setup(
    name='psm',
    version='0.1',
    description='PBSHM_mdof anomaly detection using spectra and deep learning ',
    author='Yacine Bel-Hadj',
    author_email='yacine.be-hadj@vub.be',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'pytorch-lightning',
    ],

)