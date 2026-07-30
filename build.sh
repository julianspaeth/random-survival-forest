#!/bin/sh

rm -rf random_survival_forest.egg-info build dist
uv build
uv publish
