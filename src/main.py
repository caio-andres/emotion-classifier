import os
import pickle
from keras.api.models import load_model
from src.train import train_model
from src.predict import predict_emotion


def main():
    model_path = "models/model.keras"
    tokenizer_path = "models/tokenizer.pkl"
    label_path = "models/label_encoder.pkl"

    if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_path):
        print("Model loaded from models/model.keras")
        model = load_model(model_path)
        with open(tokenizer_path, "rb") as f:
          tokenizer = pickle.load(f)
        with open(label_path, "rb") as f:
          le = pickle.load(f)
    else:
        print("Training model from scratch...")
        model, tokenizer, le = train_model()

    print("Type a phrase to test the program (or 'exit'):")

    while True:
        text = input("Text: ")
        if text.lower() == "exit":
            break
        emotion = predict_emotion(model, tokenizer, le, text)
        print(f"Emotion detected: {emotion}")


if __name__ == "__main__":
    main()
