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
    --batchSize 6 \
    --n_codes 2048 \
    --data_path "Data/ProgressiveTransformersSLP" \
    --vocab_file "Data/ProgressiveTransformersSLP/src_vocab.txt" \
    --resume_ckpt "" \
    --default_root_dir "/Dataset/everybody_sign_now_experiments/pose2text_logs/stage1" \
    --max_steps 300000 \
    --max_frames_num 200 \
    --gpu_ids "0" \
    --backmodel "/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/heatmap_model/checkpoints/epoch=7-step=28383-val_wer=0.6861.ckpt" \
    --backmodel_hparams_file "/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/heatmap_model/hparams.yaml" \
    --backmodel2 "/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/joint_model/checkpoints/epoch=13-step=8287-val_wer=0.5971.ckpt" \
    --backmodel_hparams_file2 "/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/joint_model/hparams.yaml" \
