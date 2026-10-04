#!/usr/bin/env bash
SERVER_CONFIG_DIR="./server"

furiosa-perf run \
  --server-config ${SERVER_CONFIG_DIR}/exaone4.yaml \
  --preset baseline/text-gen
