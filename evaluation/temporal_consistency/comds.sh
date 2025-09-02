# 同时使用两个模型（默认）
python temporal_consistency_unified.py /path/to/images

# 或者明确指定使用两个模型
python temporal_consistency_unified.py /path/to/images --model both

# 仍然支持单独使用某个模型
python temporal_consistency_unified.py /path/to/images --model dino
python temporal_consistency_unified.py /path/to/images --model clip


python temporal_consistency_unified.py /path/to/images
python temporal_consistency_unified.py /root/autodl-tmp/VideoStyleID/output/playground
python style_similarity.py /root/autodl-tmp/VISION/output/exp_5/stylized_anchor_frame/0_45 /root/autodl-tmp/VISION/data/styles/tsn/the_starry_night.png --model both
python style_similarity.py /root/autodl-tmp/VISION/output/exp_5/stylized_anchor_frame/0_45 /root/autodl-tmp/VISION/data/styles/wave/ /root/autodl-tmp/VISION/data/styles/tsn/the_starry_night.png --model both


python temporal_consistency_unified.py /root/autodl-tmp/VISION/output_contrast/ccpl/exp_2
python temporal_consistency_unified.py /root/autodl-tmp/VISION/output/exp_5/stylized_video_gamma_0_45/0_34

python temporal_consistency_unified.py /root/autodl-tmp/VISION/output/baseline/bird_forest_default
python temporal_consistency_unified.py /path/to/images