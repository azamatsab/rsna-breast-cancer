import os
import shutil
import glob

ROOT = "can_cam_pos"
FODLER = "cam_pos"

os.makedirs(ROOT, exist_ok=True)

files = glob.glob(os.path.join(FODLER, "*/*png"), recursive=True)

for path in files:
    folder, name = os.path.split(path)
    folder = folder.split("/")[-1]
    if folder == name[:-4]:
        shutil.copy(path, os.path.join(ROOT, name))
