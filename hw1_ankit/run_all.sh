#!/bin/bash

for s in {0..49};
do
  for p in orig perm;
  do
    ./run_mymodel.sh $s $p
  done
done
