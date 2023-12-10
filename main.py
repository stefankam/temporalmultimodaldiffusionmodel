from load_and_preprocess_image import load_and_preprocess_image
from load_and_preprocess_text import preprocess_text
from image_branch import image_branch
from text_branch import text_branch
# Import statements (use corrected import statements)
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


image_data_1 = load_and_preprocess_image('./data/MRI_78_AD.jpeg')
print('image_data_1: ',image_data_1)
image_data_2 = load_and_preprocess_image('./data/MRI_79_CN.jpeg')

text_data_1 = preprocess_text("AD diagnosed patient 78 years old with MRI axial taken on 6/1/2006")
print('text_data_1: ', text_data_1)
text_data_2 = preprocess_text("Control subject 79 year old with MRI axial taken on 5/23/2006")

# Create sequences (assuming two time steps)
image_sequence = [image_data_1, image_data_2]
text_sequence = [text_data_1, text_data_2]

max_sequence_length =  10

# Define input shapes
image_input = Input(shape=(256, 256, 1))  # Modify according to your image dimensions
text_input = Input(shape=(max_sequence_length,))  # Modify based on your text preprocessing

# Reshape image_sequence to 3D (batch_size, time_steps, input_features)
print("Before reshaping image_sequence:")
print(image_sequence)
image_sequence = np.array(image_sequence)  # Convert to numpy array
image_sequence = np.expand_dims(image_sequence, axis=-1)  # Add a dimension for input features
print("After reshaping image_sequence:")
print(image_sequence)

# Assuming you have defined max_sequence_length
print("Before padding text_sequence:")
print(text_sequence)
#text_sequence = pad_sequences(text_sequence, padding='post', truncating='post')
print("After padding text_sequence:")
print(text_sequence)


input_features = 2  # Modify this based on your actual number of features

# Temporal modeling using LSTM
lstm_layer = LSTM(128)

# Combine image and text branches
image_output = image_branch(image_input)
text_output = text_branch(text_input)

print("Image output shape:")
print(image_output.shape)
print("Text output shape:")
print(text_output.shape)

# Concatenate the multimodal features
merged_output = concatenate([image_output, text_output])
# Assuming merged_output has shape (batch_size, input_features)
merged_output = Reshape((1, input_features))(merged_output)

# Apply LSTM for temporal modeling
temporal_output = lstm_layer(merged_output)
print("Temporal output shape:")
print(temporal_output.shape)


# Output layer
output = Dense(1, activation='sigmoid')(temporal_output)

# Create the model
model = Model(inputs=[image_input, text_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model training configuration
print("Model Training Configuration:")
print("Optimizer:", model.optimizer)
print("Loss Function:", model.loss)
print("Metrics:", model.metrics)

