import os
from rembg import remove, new_session
from PIL import Image
import cv2
from color_detector import print_color_report
from classify_garment import classify_garment
from remove_background import remove_background

def segment_and_crop(input_path, output_path, section="upper"):
    """
    Runs cloth segmentation, then crops to just the requested section.
    section can be "upper", "lower", or "full".
    """
    session = new_session("u2net_cloth_seg")
    input_image = Image.open(input_path)
    segmented = remove(input_image, session=session)

    temp_path = "temp_stacked.png"
    segmented.save(temp_path)

    stacked_image = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)

    # The segmented output is always 3 sections stacked vertically,
    # so each section is exactly one third of the total height.
    # Calculated per photo instead of hardcoded, because every photo
    # has a different original height.
    total_height = stacked_image.shape[0]
    section_height = total_height // 3

    section_order = {"upper": 0, "lower": 1, "full": 2}
    section_index = section_order[section]

    start_row = section_index * section_height
    end_row = start_row + section_height

    cropped = stacked_image[start_row:end_row, :, :]
    cv2.imwrite(output_path, cropped)

    print(f"Cropped '{section}' section saved to: {output_path}")
    return output_path


def decide_section(image_path):
    """
    Uses CLIP to decide whether the photo shows a top or a bottom,
    so we know which section of the segmentation output to crop.
    """
    labels = [
        "a photo of a top or shirt",
        "a photo of pants or a skirt",
    ]
    results = classify_garment(image_path, labels)
    top_guess = results[0][0]

    if "top or shirt" in top_guess:
        return "upper"
    else:
        return "lower"

def analyze_garment(input_path, is_flatlay=False, section=None):
    """
    Full pipeline. If is_flatlay is True, uses plain background removal
    (no person expected in frame). Otherwise, uses cloth segmentation
    to isolate the garment from a person's photo.
    """
    if is_flatlay:
        cropped_path = "garment_cropped.png"
        remove_background(input_path, cropped_path)
    else:
        if section is None:
            section = decide_section(input_path)
            print(f"Auto-detected section: {section}")

        cropped_path = "garment_cropped.png"
        segment_and_crop(input_path, cropped_path, section=section)

    print()
    print_color_report(cropped_path)


if __name__ == "__main__":
    analyze_garment("test_images/test_06_top1.jpg", is_flatlay=True)
    folder = "test_images"
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()

    for filename in files:
        path = os.path.join(folder, filename)
        print(f"=== {filename} ===")
        analyze_garment(path)
        print()