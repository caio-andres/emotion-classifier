from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


def preprocess_texts(texts: str, vocab_size=5000, max_len=100):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    return padded, tokenizer
