try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='eQTLseq',
    version='17.02.10',
    description='Hierarchical probabilistic models for multiple gene/variant associations based on NGS data',
    long_description='Hierarchical probabilistic models for multiple gene/variant associations based on NGS data',
    author='Dimitrios V. Vavoulis',
    author_email='Dimitris.Vavoulis@ndcls.well.ox.ac.uk',
    url='https://github.com/dvav/eQTLseq',
    license='MIT',
    platforms=['UNIX'],
    install_requires=['numpy>=1.12.0', 'scipy>=0.18.1', 'rpy2>=2.8.5', 'tqdm>=4.11.2'],
    packages=['eQTLseq'],
    keywords='eQTL Bayesian RNA-seq',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ]
)
