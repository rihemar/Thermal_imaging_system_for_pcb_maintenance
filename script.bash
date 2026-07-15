#!/bin/bash
cd ./CameraRGB
../CameraThermique/examples/build/PCB_extraction_lite &
PID1=$!

python3 main.py &
PID2=$!

trap "kill $PID1 $PID2" INT

wait
