import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https\S+|http\S+', '', text).strip()
    text = re.sub(r'@\S+', '', text).strip()
    text = re.sub(r'[^a-z0-9\s]', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def prepare_data(data):
    data['text_clean'] = data['text'].apply(clean_text)

    dropna = data[data.text_clean == '']['text_clean'].count()
    
    if dropna != 0:
        print(f'Удалено {dropna} пропусков.')
        data = data[data.text_clean != '']

    data[['text_clean']].to_csv('data/dataset_processed.csv', index=False)
    print('Датасет предобработан.')


def train_test_val():
    data = pd.read_csv('data/dataset_processed.csv')

    train, test_val = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    val, test = train_test_split(test_val, test_size=0.5, random_state=RANDOM_STATE)

    train.to_csv('data/train.csv', index=False)
    val.to_csv('data/val.csv', index=False)
    test.to_csv('data/test.csv', index=False)

    print('Разделение на трейн, валидацию и тест прошло успешно.')
    print('Train:', train.shape)
    print('Val:', val.shape)
    print('Test:', test.shape)