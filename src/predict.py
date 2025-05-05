from keras.api.models import load_model
from keras_preprocessing.sequence import pad_sequences

def predict_emotion(model, tokenizer, label_encoder, text):
  seq = tokenizer.texts_to_sequences([text])
  padded = pad_sequences(seq, maxlen=100, padding="post", truncating="post")
  prediction = model.predict(padded)
  class_index = prediction.argmax(axis=1)[0]
  emotion_label = label_encoder.inverse_transform([class_index])[0]
  return emotion_label