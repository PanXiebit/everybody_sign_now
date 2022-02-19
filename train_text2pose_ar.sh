python -m train_text2pose_ar \
    --gpus 2 \
    --batchSize 4 \
    --data_path "/data/xp_data/slr/EverybodySignNow/Data/how2sign" \
    --text_path "data/text2gloss/" \
    --vocab_file "data/text2gloss/how2sign_vocab.txt" \
    --pose_vqvae "logs/SeqLen_{16}_TemDs_{1}_AttenType_{spatial-temporal-joint}_DecoderType_{divided-unshare}/lightning_logs/version_0/checkpoints/epoch=16-step=50965.ckpt" \
    --hparams_file "logs/SeqLen_{16}_TemDs_{1}_AttenType_{spatial-temporal-joint}_DecoderType_{divided-unshare}/lightning_logs/version_0/hparams.yaml" \
    --resume_ckpt "" \
    --default_root_dir "text2pose_logs/separate_slice" \
    --max_steps 300000 \
    --max_frames_num 400 \
    --gpu_ids "0,1" \


    # --data_path /Dataset/how2sign/video_and_keypoint \
    # --sequence_length 4 \
    # --debug 0

# python -m train_singlepose_recon \
#     --gpus 1 \
#     --batchSize 128 \
#     --data_path /Dataset/how2sign/video_and_keypoint \
#     --sequence_length 1 \
#     --debug 0