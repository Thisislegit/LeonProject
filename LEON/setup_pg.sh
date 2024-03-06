#!/bin/bash

# Check if ./log/ exists, if not create it
if [ ! -d "./log/" ]; then
    mkdir "./log/"
fi

# Check if ./logs/ exists, if not create it
if [ ! -d "./logs/" ]; then
    mkdir "./logs/"
fi

# Set the installation directory of PostgreSQL
PREFIX_DIR="/data1/chenx/projects/pg_install_ml_5438/"

# install PostgreSQL
./configure --prefix="$PREFIX_DIR"

make clean
make -j
make install

# 进入目录
cd "$PREFIX_DIR"

# Restart PostgreSQL
./bin/pg_ctl -D ./data restart
./bin/pg_ctl -D ./data1 restart
./bin/pg_ctl -D ./data2 restart

# Prewarm the database
cd "/data1/chenx/projects/LeonOpenSource/LEON/conf"
python pre_warm.py
