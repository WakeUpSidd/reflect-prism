import os
from pathlib import Path
from PIL import Image
import numpy as np

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/Siddhant/data"))

# Trans10K
p_img = DATA_ROOT/"trans10k_export_bin/train/rgb/000000.png"
p_msk = DATA_ROOT/"trans10k_export_bin/train/mask/000000.png"
img = Image.open(p_img).convert("RGB")
msk = np.array(Image.open(p_msk))
print("Trans10K img", img.size, "mask", msk.shape, "unique", np.unique(msk))

# NYUv2
p_rgb = DATA_ROOT/"nyuv2_export/train/rgb/000000.png"
p_dep = DATA_ROOT/"nyuv2_export/train/depth/000000.npy"
p_nrm = DATA_ROOT/"nyuv2_export/train/normal/000000.npy"
rgb = Image.open(p_rgb).convert("RGB")
dep = np.load(p_dep)
nrm = np.load(p_nrm)
print("NYUv2 rgb", rgb.size, "depth", dep.shape, dep.dtype, "normal", nrm.shape, nrm.dtype)
