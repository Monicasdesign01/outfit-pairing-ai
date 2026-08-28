import cv2

def crop_upper_body(input_path, output_path, section_height=1350):
    # Load the full stacked image (upper + lower + full body, one below another)
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Keep only the first section_height rows — the "upper body" section,
    # since u2net_cloth_seg always stacks upper/lower/full in that order
    upper_section = image[0:section_height, :, :]

    cv2.imwrite(output_path, upper_section)
    print(f"Saved cropped upper section to: {output_path}")


if __name__ == "__main__":
    crop_upper_body("test2_clothseg.png", "test2_upper_only.png")