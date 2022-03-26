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



python -m train_text2pose \
    --gpus 1 \
    --batchSize 8 \
    --data_path "Data/ProgressiveTransformersSLP" \
    --vocab_file "Data/ProgressiveTransformersSLP/src_vocab.txt" \
    --pose_vqvae "logs/phoneix_spl_seperate_SeqLen_1_5120/lightning_logs/version_3/checkpoints/epoch=123-step=50095.ckpt" \
    --vqvae_hparams_file "logs/phoneix_spl_seperate_SeqLen_1_5120/lightning_logs/version_3/hparams.yaml" \
    --resume_ckpt "" \
    --default_root_dir "/Dataset/everybody_sign_now_experiments/text2pose_logs/ctc_vnat" \
    --max_steps 300000 \
    --max_frames_num 200 \
    --gpu_ids "0" \
    --ctc_model "pose2text_logs/lightning_logs/version_1/checkpoints/epoch=28-step=1623.ckpt" \
    --ctc_hparams_file "pose2text_logs/lightning_logs/version_1/hparams.yaml" \


# python -m train_text2pose \
#     --gpus 1 \
#     --batchSize 8 \
#     --data_path "Data/ProgressiveTransformersSLP" \
#     --vocab_file "Data/ProgressiveTransformersSLP/src_vocab.txt" \
#     --pose_vqvae "/Dataset/everybody_sign_now_experiments/text2pose_logs/ctc_nat/lightning_logs/freeze_emb/checkpoints/epoch=98-step=87812.ckpt" \
#     --vqvae_hparams_file "/Dataset/everybody_sign_now_experiments/text2pose_logs/ctc_nat/lightning_logs/freeze_emb/hparams.yaml" \
#     --resume_ckpt "" \
#     --default_root_dir "/Dataset/everybody_sign_now_experiments/text2pose_logs/nat_distill" \
#     --max_steps 300000 \
#     --max_frames_num 200 \
#     --gpu_ids "0" \
#     --ctc_model "pose2text_logs/lightning_logs/version_1/checkpoints/epoch=28-step=1623.ckpt" \
#     --ctc_hparams_file "pose2text_logs/lightning_logs/version_1/hparams.yaml" \