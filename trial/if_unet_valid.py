# 快速验证脚本
from diffusers import UNet2DConditionModel

# 确保加载的是 SD 1.5 的 UNet
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

print(f"Number of up_blocks: {len(unet.up_blocks)}") # 会打印 4
print(unet.up_blocks[3]) # 不会报错