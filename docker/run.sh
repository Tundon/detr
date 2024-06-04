#!/bin/bash

docker run -itd --rm --name detr \
  --gpus all \
  -v .:/workspace/detr \
  -v /mnt/e/Data/COCO-2017:/workspace/data/coco2017 \
  teabots/pytorch:2.3.0 \
  /bin/bash
