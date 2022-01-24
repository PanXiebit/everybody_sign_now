python -m train_text2pose \
    --gpus 1 \
    --batchSize 5 \
    --pose_vqvae "lightning_logs/seqlen_1_heuristic_downsample_lr_decay_v2/checkpoints/epoch=37-step=752323.ckpt" \
    
    # --data_path /Dataset/how2sign/video_and_keypoint \
    # --sequence_length 4 \
    # --debug 0

# python -m train_singlepose_recon \
#     --gpus 1 \
#     --batchSize 128 \
#     --data_path /Dataset/how2sign/video_and_keypoint \
#     --sequence_length 1 \
#     --debug 0