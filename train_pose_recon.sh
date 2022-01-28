
atten_type="spatial-only"
decoder_type="joint"
temporal_downsample=1
sequence_length=4

python -m train_pose_recon \
    --gpus 1 \
    --batchSize 96 \
    --data_path /data/xp_data/slr/EverybodySignNow/Data/how2sign \
    --n_codes 1024 \
    --sequence_length ${sequence_length} \
    --temporal_downsample ${temporal_downsample} \
    --atten_type ${atten_type} \
    --decoder_type ${decoder_type} \
    --default_root_dir "logs/SeqLen_{${sequence_length}}_TemDs_{${temporal_downsample}}_AttenType_{${atten_type}}_DecoderType_{${decoder_type}}" \
    --debug 0 \
    --gpu_ids "1"

# python -m train_singlepose_recon \
#     --gpus 1 \
#     --batchSize 128 \
#     --data_path /Dataset/how2sign/video_and_keypoint \
#     --sequence_length 1 \
#     --debug 0