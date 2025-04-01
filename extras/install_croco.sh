#!/bin/bash

SETUP_PATH="./submodules/croco/setup.py"

cat > "$SETUP_PATH" <<EOF
from setuptools import setup, find_packages

setup(
    name="croco",
    version="0.1",
    packages=find_packages(where="croco"),
    package_dir={"": "croco"},
)
EOF

touch submodules/croco/croco/__init__.py
pip install -e submodules/croco/ --config-settings editable_mode=compat
