from rembg import remove, new_session
from PIL import Image

def remove_background(input_path, output_path):
    session = new_session("u2netp")

    input_image = Image.open(input_path)
    output_image = remove(input_image, session=session)
    output_image.save(output_path)
    print(f"Saved background-removed image to: {output_path}")


if __name__ == "__main__":
    remove_background("test2.jpg", "test2_nobg.png")