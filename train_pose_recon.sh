# python -m train_pose_recon \
#     --gpus 1 \
#     --batchSize 256 \
#     --data_path /Dataset/how2sign/video_and_keypoint \
#     --sequence_length 4 \
#     --debug 0

python -m train_singlepose_recon \
    --gpus 1 \
    --batchSize 128 \
    --data_path /Dataset/how2sign/video_and_keypoint \
    --sequence_length 1 \
    --debug 0