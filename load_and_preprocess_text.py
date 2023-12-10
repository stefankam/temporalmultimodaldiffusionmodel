from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(text, tokenizer=None, max_sequence_length=None, vocab_size=None):
    """
    Preprocess text data by tokenizing and converting to numerical sequences.

    Parameters:
    - text: str, the input text to be preprocessed.
    - tokenizer: Tokenizer object (optional), if provided, it will be used for tokenization.
    - max_sequence_length: int (optional), maximum sequence length for padding.
    - vocab_size: int (optional), maximum vocabulary size for tokenization.

    Returns:
    - Preprocessed numerical sequence.
    - Tokenizer object (if not provided).
    """
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts([text])

    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])

    if max_sequence_length is not None:
        sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    return sequences, tokenizer

# Example usage:
text = "This is a sample text for preprocessing."
max_sequence_length = 10
vocab_size = 10000

preprocessed_text, tokenizer = preprocess_text(text, max_sequence_length=max_sequence_length, vocab_size=vocab_size)
print("Preprocessed Text:")
print(preprocessed_text)
print("Tokenizer Word Index:")
print(tokenizer.word_index)
