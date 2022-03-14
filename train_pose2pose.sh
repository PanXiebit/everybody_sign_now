
atten_type="spatial-temporal-joint"
decoder_type="divided-unshare"
temporal_downsample=1
sequence_length=16
n_codes=5120

python -m train_pose2pose \
    --gpus 1 \
    --batchSize 128 \
    --data_path Data/how2sign \
    --n_codes ${n_codes} \
    --sequence_length ${sequence_length} \
    --debug 0 \
    --gpu_ids "0" \
    --default_root_dir "logs/spl_SeqLen_${sequence_length}" \
    # --default_root_dir "logs/test" \


# python -m train_singlepose_recon \
#     --gpus 1 \
#     --batchSize 128 \
#     --data_path /Dataset/how2sign/video_and_keypoint \
#     --sequence_length 1 \
#     --debug 0