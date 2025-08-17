Example 1: Process all videos with all styles


python video_style_transfer.py \
    --content_path ./data/content \
    --style_path ./data/style \
    --output_dir ./video_results \
    --ddim_steps 50


The results will be saved in video_results/ with folders named like:
video_results/van_gogh_stylized_video1/
video_results/van_gogh_stylized_video2/
video_results/monet_stylized_video1/
video_results/monet_stylized_video2/
Example 2: Process a single video with a single style


python video_style_transfer.py \
    --content_path ./data/content/video1 \
    --style_path ./data/style/van_gogh.jpg \
    --output_dir ./video_results



This will create one output folder: video_results/van_gogh_stylized_video1/.

Example 3: Process all videos with a single style


python video_style_transfer.py \
    --content_path ./data/content \
    --style_path ./data/style/monet.png \
    --output_dir ./video_results \
    --gamma 0.6 --temperature 1.2


This will create video_results/monet_stylized_video1/ and video_results/monet_stylized_video2/.