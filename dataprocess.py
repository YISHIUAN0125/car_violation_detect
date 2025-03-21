import os
import shutil
import random

# set the source and target directories
source_dir = "img_preprocess"
target_dir = "yolo_dataset"
os.makedirs(target_dir, exist_ok=True)

# determine the split ratio
splits = {"train": 0.8, "valid": 0.1, "test": 0.1}

# build yolo directory structure
for split in splits.keys():
    os.makedirs(os.path.join(target_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, split, "labels"), exist_ok=True)

# fetch all images
image_exts = (".jpg", ".jpeg", ".png")
all_images = [f for f in os.listdir(source_dir) if f.endswith(image_exts)]

# suffle data
random.shuffle(all_images)

total = len(all_images)
split_sizes = {key: int(total * ratio) for key, ratio in splits.items()}

# Make sure the sum of split sizes equals to the total number of images
split_sizes["train"] += total - sum(split_sizes.values())

# sploit data
start = 0
for split, size in split_sizes.items():
    end = start + size
    subset = all_images[start:end]
    start = end
    
    for img_file in subset:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        
        img_src = os.path.join(source_dir, img_file)
        label_src = os.path.join(source_dir, label_file)
        
        img_dst = os.path.join(target_dir, split, "images", img_file)
        label_dst = os.path.join(target_dir, split, "labels", label_file)
        
        shutil.copy(img_src, img_dst)
        if os.path.exists(label_src):  # Make sure label file exists
            shutil.copy(label_src, label_dst)

print("data saved at yolo_dataset/")