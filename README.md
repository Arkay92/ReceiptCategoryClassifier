# ReceiptCategoryClassifier

An AI-powered tool that leverages TensorFlow and NLP techniques to classify receipt categories based on OCR-processed text.

## Overview

ReceiptCategoryClassifier is designed to simplify the task of categorizing receipts, making it an essential tool for automating expense tracking and financial organization. By analyzing the text extracted from receipt images, this tool predicts the most likely category of each receipt, such as groceries, utilities, dining, etc.

## Features

- **Text Preprocessing**: Incorporates various NLP techniques to clean and prepare OCR text for classification.
- **Deep Learning Model**: Utilizes a Convolutional Neural Network (CNN) architecture for accurate text classification.
- **CLI Support**: Offers a straightforward Command-Line Interface for training the model and predicting categories.
- **Batch Prediction**: Capable of processing multiple texts at once for efficient bulk categorization.

## Getting Started

### Prerequisites

Ensure you have Python 3.6 or later installed. You will also need the following packages:

```sh
tensorflow
numpy
nltk
scikit-learn
datasets
```

Install them using the following command:

```
pip install -r requirements.txt
```
## Usage
### Training the Model

Train the model with your dataset:

```
python main.py --train
```
### Predicting Categories

Predict the category of a single receipt:
```
python main.py --predict "Your OCR receipt text here"
```

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check issues page if you want to contribute.

## License
Distributed under the MIT License. See LICENSE for more information.
