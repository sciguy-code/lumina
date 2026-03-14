import os
import glob
from lumina.io.loader import load_image
from lumina.io.saver import save_image
from lumina.pipeline.engine import Pipeline
from lumina.filters.basic import sharpen, adjust_contrast, adjust_brightness
from lumina.filters.enchant import dreamglow, vignette, vaporwave

# set input and output directories
input_dir = "samples"
output_dir = "enchanting_images"

# make sure we have somewhere to put the results
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# grab all the test images
image_files = glob.glob(os.path.join(input_dir, "*.*"))
image_files = [f for f in image_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

# make sure we found them
if not image_files:
    print("uh oh, couldn't find the test images!")
    exit(1)

print(f"Found {len(image_files)} images, starting processing.")

# build the primary image processing pipeline
dream_pipeline = Pipeline()
dream_pipeline.add(vaporwave)
dream_pipeline.add(dreamglow, intensity=0.7)
dream_pipeline.add(sharpen)
dream_pipeline.add(vignette, strength=1.2)
dream_pipeline.add(adjust_contrast, factor=1.1)

# build a secondary image processing pipeline
pop_pipeline = Pipeline()
pop_pipeline.add(adjust_brightness, factor=1.1)
pop_pipeline.add(adjust_contrast, factor=1.3)
pop_pipeline.add(sharpen)
pop_pipeline.add(dreamglow, intensity=0.5)

# process each image to generate 2 variations
for idx, file_path in enumerate(image_files):
    print(f"processing {os.path.basename(file_path)}...")
    img = load_image(file_path)

    # generate dreamwave version
    dream_result = dream_pipeline.run(img)
    dream_path = os.path.join(output_dir, f"enchanted_dream_{idx}.png")
    save_image(dream_result, dream_path)

    # generate hyperpop version
    pop_result = pop_pipeline.run(img)
    pop_path = os.path.join(output_dir, f"enchanted_pop_{idx}.png")
    save_image(pop_result, pop_path)

print("all done! check the enchanting_images folder.")
