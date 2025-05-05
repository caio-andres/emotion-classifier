from src.train import train_model
from src.predict import predict_emotion


def main():
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
