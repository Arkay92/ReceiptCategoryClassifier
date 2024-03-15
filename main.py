import re
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from datasets import load_dataset
import logging
import argparse
import nltk
import os
import pickle

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(texts):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stemmer = nltk.stem.PorterStemmer()

    processed_texts = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = nltk.word_tokenize(text)
        text = ' '.join([stemmer.stem(word) for word in words if word not in stop_words])
        processed_texts.append(text)
    return processed_texts

def train_and_save_model():
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v1")
    train_texts = preprocess_text(dataset['train']['raw_data'])
    val_texts = preprocess_text(dataset['valid']['raw_data'])

    train_labels = [example['category'] for example in dataset['train']['parsed_data']]
    val_labels = [example['category'] for example in dataset['valid']['parsed_data']]

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    val_labels = label_encoder.transform(val_labels)

    # Save the label encoder for later use in prediction
    with open('label_encoder.pkl', 'wb') as le_file:
        pickle.dump(label_encoder, le_file)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    class_weights_dict = dict(enumerate(class_weights))

    max_features = 20000
    max_len = 128
    vectorize_layer = TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=max_len)
    vectorize_layer.adapt(train_texts)

    model = Sequential([
        vectorize_layer,
        Embedding(max_features + 1, 64, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(train_labels)), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])

    train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_labels)).batch(32)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model.fit(train_ds, epochs=10, validation_data=val_ds, class_weight=class_weights_dict, callbacks=[early_stopping])
    model.save('receipt_classifier_model')

def predict_receipt_category(receipt_text):
    processed_text = preprocess_text([receipt_text])

    if not os.path.exists('receipt_classifier_model'):
        logging.error("Model file not found. Train the model first using --train.")
        return

    model = tf.keras.models.load_model('receipt_classifier_model')

    if not os.path.exists('label_encoder.pkl'):
        logging.error("Label encoder file not found. Train the model first using --train.")
        return

    with open('label_encoder.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)

    prediction = model.predict(processed_text)
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))

    print(f"Predicted Category: {predicted_label[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receipt Category Classifier")
    parser.add_argument('--train', action='store_true', help="Train the model and save it")
    parser.add_argument('--predict', type=str, help="Predict the category of a receipt given its text")

    args = parser.parse_args()

    if args.train:
        train_and_save_model()
    elif args.predict:
        predict_receipt_category(args.predict)
    else:
        parser.print_help()
