#!/bin/sh

rm -r random_survival_forest.egg-info
rm -r build
rm -r dist
python -m pip install --user --upgrade setuptools wheel
python -m pip install --user --upgrade twine
python setup.py sdist bdist_wheel
python -m twine upload dist/*
