# python -m train_text2pose \
#     --gpus 1 \
#     --batchSize 5 \
#     --data_path "Data/how2sign" \
#     --text_path "data/text2gloss/" \
#     --vocab_file "data/text2gloss/how2sign_vocab.txt" \
#     --pose_vqvae "logs/phoneix_spl_seperate_SeqLen_1/lightning_logs/version_3/checkpoints/epoch=123-step=50095.ckpt" \
#     --hparams_file "logs/phoneix_spl_seperate_SeqLen_1/lightning_logs/version_3/hparams.yaml" \
#     --resume_ckpt "" \
#     --default_root_dir "text2pose_logs/test" \
#     --max_steps 300000 \
#     --max_frames_num 200 \
#     --gpu_ids "0" \



python -m train_pose2text \
    --gpus 1 \
    --batchSize 128 \
    --data_path "Data/ProgressiveTransformersSLP" \
    --text_path "data/text2gloss/" \
    --vocab_file "Data/ProgressiveTransformersSLP/src_vocab.txt" \
    --resume_ckpt "" \
    --default_root_dir "pose2text_logs/" \
    --max_steps 300000 \
    --max_frames_num 200 \
    --gpu_ids "0" \