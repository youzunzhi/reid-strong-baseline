# Experiment all tricks without center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on
# Dataset 2: dukemtmc
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('dukemtmc')" OUTPUT_DIR "('/home/zunzhi/reid-strong-baseline/log/reid_baseline_review/Opensource_test/dukemtmc/Experiment-all-tricks-256x128-bs16x4-warmup10-erase0-labelsmooth_on-laststride1-bnneck_on')"