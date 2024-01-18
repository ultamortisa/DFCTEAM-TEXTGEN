import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

model_dir = "pretrained"

with open(f'./{model_dir}/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = json.loads(f.read())
    tokenizer = tokenizer_from_json(tokenizer_json)

model = tf.keras.models.load_model(f'./{model_dir}/model')
max_sequence_len = int(open(f"./{model_dir}/max_sequence_len.txt", "r", encoding="utf-8").read())

while True:
    seed_text = input("prompt> ")
    next_words = int(input("How many words are we going to generate? (int)> "))

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probabilities = model.predict(token_list, verbose=0)[0]
        predicted = np.argmax(predicted_probabilities)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        print(output_word, end=" ", flush=True)

    print("\n")