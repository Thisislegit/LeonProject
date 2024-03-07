#!/bin/bash

# Check if ./log/ exists, if not create it
if [ ! -d "./log/" ]; then
    mkdir "./log/"
fi

# Check if ./logs/ exists, if not create it
if [ ! -d "./logs/" ]; then
    mkdir "./logs/"
fi

PREFIX_DIR="/data1/chenx/projects/pg_install_ml_5438"

cd "$PREFIX_DIR"

# restart PostgreSQL
./bin/pg_ctl -D ./data restart
./bin/pg_ctl -D ./data1 restart
./bin/pg_ctl -D ./data2 restart


cd "/data1/chenx/projects/LeonOpenSource/LEON/conf"
python pre_warm.py

cd ..


# Delete unused checkpoint and log files
cd log
rm model.pth
rm messages.pkl
cd ..

# Start the server
python leon_server.py

