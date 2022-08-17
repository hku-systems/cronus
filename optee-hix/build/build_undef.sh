#/bin/bash

sudo make optee-os -j40 &> error.log.raw
grep -i "undefined reference to" <error.log.raw > error.log
python3 find_all.py > error.log.strip
cat error.log.strip | sort |uniq > error
