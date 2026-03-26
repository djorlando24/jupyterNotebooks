#!/bin/bash
rm -r build
rm Python_Library/ArrayBin_Cython.c
python3 setup.py build_ext --inplace
