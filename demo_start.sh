export LD_LIBRARY_PATH="/home/user/.conda/envs/python_3.9/lib/"

python ./tools/demo.py --cfg_file ./tools/cfgs/kitti_models/pointpillar.yaml --ckpt ./tools/pointpillar_7728.pth --data_path ./tools/000001.bin
