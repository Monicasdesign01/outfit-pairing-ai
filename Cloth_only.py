from rembg import remove, new_session
from PIL import Image

session = new_session("u2net_cloth_seg")

input_image = Image.open("test2.jpg")
output_image = remove(input_image, session=session)
output_image.save("test2_clothseg.png")

print("Saved to test2_clothseg.png")