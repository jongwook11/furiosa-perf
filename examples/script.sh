#!/usr/bin/env bash
BENCH_CONFIG_DIR="./bench"
SERVER_CONFIG_DIR="./server"

furiosa-perf run \
  --server-config ${SERVER_CONFIG_DIR}/exaone4.yaml \
  --benchmark-config ${BENCH_CONFIG_DIR}/llm_scenario.yaml
