# Emotion Classification Tool with Keras, NLP and Deep Learning

A Python tool that classifies emotions in short text phrases using NLP, Deep Learning, and data from Hugging Face. It automates the full process from training to prediction using Keras and Scikit-learn.

## What you need

- Python 3.x (latest version recommended)
- pip (Python package manager)
- Internet access (to download the Hugging Face dataset)

## Getting started

1. First, clone this repository:
```bash
git clone https://github.com/caio-andres/emotion-classifier.git
cd emotion-classifier
```

2. Create and activate your virtual environment:
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate
```

3. Install all dependencies:
```bash
pip install -r requirements.txt
```

## How it works

This tool will automatically:

- Download the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset from Hugging Face 🤗
- Preprocess and tokenize the dataset using `keras_preprocessing`
- Encode the labels using `LabelEncoder` from scikit-learn
- Train a model using Keras (Deep Learning)
- Save the model, tokenizer, and label encoder using `pickle`
- Enter interactive mode for live predictions

## Project structure

```
emotion-classifier/
├── models/                    # Saved model and encoders
│   ├── model.keras            # Trained model file
│   ├── tokenizer.pkl          # Tokenizer used to process input
│   └── label_encoder.pkl      # LabelEncoder for emotion labels
├── src/                       # Source code
│   ├── main.py                # Program entry point (automation starts here)
│   ├── train.py               # Training and model saving
│   ├── predict.py             # Prediction logic
│   └── preprocess.py          # Text preprocessing (tokenizing, padding)
├── requirements.txt           # Dependencies list
└── README.md                  # This file
```

## How to use

1. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate
```

2. Run the app:
```bash
python -m src.main
```

3. Example usage:
```bash
Text: I'm feeling great
→ Emotion detected: joy

Text: I am so frustrated
→ Emotion detected: anger
```

Type `exit` to stop.

## Technologies used

- `tensorflow` / `keras`: Deep Learning framework
- `keras-preprocessing`: Text tokenization and padding
- `scikit-learn`: Label encoding and data splitting
- `huggingface/datasets`: Real dataset with emotion labels
- `pickle`: Saving and loading models and encoders
- `pandas`: Data handling

---

Built with much effort by [Caio André](https://github.com/caio-andres) 💙