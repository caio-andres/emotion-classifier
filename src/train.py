import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Embedding, GlobalAveragePooling1D, Dense
from src.preprocess import preprocess_texts


def train_model(dataset_path="data/emotions.csv"):
    df = pd.read_csv(dataset_path)
    texts, labels = df["text"].tolist(), df["label"].tolist()
