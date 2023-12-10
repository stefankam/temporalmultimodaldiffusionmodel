from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

# Define the text processing branch
def build_text_branch(max_sequence_length, vocab_size, embedding_dim, output_dim=1):
    # Create a Sequential model for text processing
    text_model = Sequential()

    # Embedding layer
    text_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))

    # LSTM layer
    text_model.add(LSTM(128, return_sequences=True))
    text_model.add(LSTM(64))

    # Output layer
    text_model.add(Dense(output_dim, activation='linear'))  # Output layer based on your task

    return text_model

# Define the maximum sequence length, vocabulary size, and embedding dimension (customize these based on your text data)
max_sequence_length = 50  # Adjust based on your text data
vocab_size = 10000  # Adjust based on your vocabulary size
embedding_dim = 100  # Adjust based on your choice

# Build the text processing branch
text_branch = build_text_branch(max_sequence_length, vocab_size, embedding_dim)
print("text_branch: ", text_branch)