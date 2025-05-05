import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Embedding, GlobalAveragePooling1D, Dense
from src.preprocess import preprocess_texts


def train_model():
    ds = load_dataset("dair-ai/emotion", "split")

    df = pd.concat(
        [ds["train"].to_pandas(), ds["validation"].to_pandas(), ds["test"].to_pandas()]
    )
    texts, labels = df["text"].tolist(), df["label"].tolist()

    X, tokenizer = preprocess_texts(texts)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model = Sequential(
        [
            Embedding(input_dim=5000, output_dim=32, input_length=X.shape[1]),
            GlobalAveragePooling1D(),
            Dense(32, activation="relu"),
            Dense(len(set(y)), activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16
    )

    model.save("models/model.keras")
    return model, tokenizer, le
