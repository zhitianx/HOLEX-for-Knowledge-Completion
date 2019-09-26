#!/bin/bash
for i in `seq 1 32`; do
	echo DC=$i beaker experiment create -f proje_haar5.yaml
done

