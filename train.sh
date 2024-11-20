python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --lr_drop 100 --epochs 150 --lr "3e-5" --lr_backbone "3e-6" --batch_size 2 --coco_path data/coco --output_dir output > output/log.txt