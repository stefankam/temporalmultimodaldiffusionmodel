from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
import numpy as np
import pandas
import plotly.express as px
import plotly.graph_objects as go

# Function to load and preprocess the image
def load_and_preprocess_image(file_path, target_size=(256, 256)):
    image = load_img(file_path, color_mode='grayscale', target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Function to build a simple CNN model
def build_diffusion_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Load and preprocess the images
ad_image_processed = load_and_preprocess_image('./data/MRI_78_AD.jpeg')
cn_image_processed = load_and_preprocess_image('./data/MRI_79_CN.jpeg')

# Assuming the input shape for the model is the same as the target size of the images
input_shape = (256, 256, 1)  # Grayscale implies one channel
model = build_diffusion_model(input_shape)

# Model summary
model.summary()

# Create subplots with 1 row and 2 columns
# Assuming ad_image_processed and cn_image_processed are grayscale images (2D arrays)
#ad_image_processed = np.random.rand(64, 64)  # Replace with your actual data
#cn_image_processed = np.random.rand(64, 64)  # Replace with your actual data

# Create subplots with 1 row and 2 columns
fig = go.Figure()

# Add the first image on the left as a heatmap
fig.add_trace(go.Heatmap(z=np.squeeze(ad_image_processed), colorscale='gray', showscale=False, name='78 year old with AD'))

# Add the second image on the right as a heatmap
fig.add_trace(go.Heatmap(z=np.squeeze(cn_image_processed), colorscale='gray', showscale=False, name='79 year old CN'))

# Update the layout
fig.update_layout(
    title='Image Comparison',
    xaxis=dict(showline=False, showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showline=False, showgrid=False, zeroline=False, showticklabels=False),
)

# Show the plot
fig.show()
