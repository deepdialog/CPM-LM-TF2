#!/bin/bash

set -e
docker build -t qhduan/onnx-cpm-gen:0.1 .
docker push qhduan/onnx-cpm-gen:0.1

