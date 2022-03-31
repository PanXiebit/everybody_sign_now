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
    --pose_vqvae "/Dataset/everybody_sign_now_experiments/pose2text_logs/stage1/lightning_logs/joint_heatmap_seperate/checkpoints/epoch=29-step=35489-val_wer=0.0000-val_rec_loss=0.0221-val_ce_loss=0.0000.ckpt" \
    --vqvae_hparams_file "/Dataset/everybody_sign_now_experiments/pose2text_logs/stage1/lightning_logs/joint_heatmap_seperate/hparams.yaml" \
    --resume_ckpt "" \
    --default_root_dir "/Dataset/everybody_sign_now_experiments/text2pose_logs/stage2" \
    --max_steps 300000 \
    --max_frames_num 200 \
    --gpu_ids "0" \
    --backmodel2 "/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/joint_model/checkpoints/epoch=13-step=8287-val_wer=0.5971.ckpt" \
    --backmodel_hparams_file2 "/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/joint_model/hparams.yaml" \
    --backmodel "/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/heatmap_model/checkpoints/epoch=7-step=28383-val_wer=0.6861.ckpt" \
    --backmodel_hparams_file "/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/heatmap_model/hparams.yaml" \


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