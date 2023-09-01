from setuptools import setup, find_packages

setup(
    name='psm',
    version='0.1',
    description='PBSHM_mdof anomaly detection using spectra and deep learning',
    author='Yacine Bel-Hadj',
    author_email='yacine.bel-hadj@vub.be',
    packages=find_packages(include=['psm', 'psm.*']),
    package_data={'': ['config.py']},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'pytorch-lightning',
        'comet-ml',
        'dynaconf',
        'joblib',
        'scikit-learn',
        'tqdm',
        
    ],
)


