python run_full_evaluation.py \
    --baselines_dir /root/autodl-tmp/quantitative_comparison_results/batch_4/ \
    --ours_dir /root/autodl-tmp/video_style_transfer/results/quant_exp_results_batch_4 \
    --ablation_dir /root/autodl-tmp/video_style_transfer/results/ablation_styleid \
    --content_dir /root/autodl-tmp/video_style_transfer/data/data_quant/content \
    --style_dir /root/autodl-tmp/video_style_transfer/data/data_quant/style \
    --output_dir ./quant_results/ \
    --aesthetic_model_path ./improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth \
    --i3d_model_path ./frechet_video_distance/loaded_models/i3d_torchscript.pt \
    --batch_size 32