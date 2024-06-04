#!/bin/bash

docker run -itd --rm --name detr \
  --gpus all \
  -v .:/workspace/detr \
  teabots/pytorch:2.3.0 \
  /bin/bash
