from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='random_survival_forest',
    packages=['random_survival_forest'],
    version='0.1.2',
    license="MIT License",
    long_description=readme,
    long_description_content_type="text/markdown",
    description='A Random Survival Forest implementation inspired by Ishwaran et al.',
    author='Julian Späth',
    author_email='spaethju@posteo.de',
    url='https://github.com/julianspaeth/random-survival-forest',
    download_url='https://github.com/julianspaeth/random-survival-forest/archive/v0.1-beta.tar.gz',
    keywords=['survival-analysis', 'survival-prediction', 'machine-learning', 'random-forest',
              'random-survival-forest'],
    install_requires=[
        'numpy',
        'pandas',
        'joblib',
        'multiprocess',
        'lifelines',
        'scikit-learn',
        'Cython',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Software Development :: Build Tools', 'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
