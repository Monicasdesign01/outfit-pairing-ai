import os
import cv2
import numpy as np
from analyse_garment import segment_and_crop, decide_section

folder = "test_images"
files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
files.sort()

print("Checking how much real content each cropped section actually has:\n")

for filename in files:
    path = os.path.join(folder, filename)

    section = decide_section(path)
    segment_and_crop(path, "temp_check.png", section=section)

    image = cv2.imread("temp_check.png", cv2.IMREAD_UNCHANGED)
    alpha = image[:, :, 3]
    visible_percent = (alpha > 128).sum() / alpha.size * 100

    print(f"{filename}: {visible_percent:.1f}% visible ({section})")