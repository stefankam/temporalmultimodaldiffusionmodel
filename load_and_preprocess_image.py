from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Function to load and preprocess the image
def load_and_preprocess_image(file_path, target_size=(256, 256)):
    image = load_img(file_path, color_mode='grayscale', target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize to [0, 1]
    return image