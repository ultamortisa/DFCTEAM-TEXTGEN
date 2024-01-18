print("What architecture use?")
print("1 - Transformer (uses more RAM, but better than LSTM)")
print("2 - LSTM (uses less RAM, but can generate strange text)")
print("3 - LSTM with Attention (uses less RAM, but can generate strange text)")
print()
ch1 = int(input("Your choice (1, 2 or 3)> "))
if ch1 == 1:
    print("Selected 1 (Transformer)")
elif ch1 == 2:
    print("Selected 2 (LSTM)")
elif ch1 == 3:
    print("Selected 3 (LSTM with Attention)")
else:
    print("Nothing selected, calling exit()")
    exit()

print()
epochs = int(input("How much iterations (epochs) to train? (int)> "))
print()

# TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model

if ch1 == 1:
    from transformer import Transformer

import json

print("Detecting GPU...")
gpu_list = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpu_list)}")
if len(gpu_list) == 0: print("No GPU found")
else: print(f"Detected {len(gpu_list)} GPUs")
print()
print("Loading dataset.txt...")

with open('dataset.txt', 'r', encoding='utf-8') as file:
    data = file.read()

print()
print("How much lines you want use in dataset? (100 means you will use first 100 lines from dataset)")
print("More lines - needs more RAM")
print()
lines = int(input("Your input (int)> "))
sentences = data.split('\n')[:lines]#[-100:]#[:100]#[580050:580070]#[595000:596000]#[:100]#[:30000]#[::4000]
print(f"len(sentences): {len(sentences)}")
print("tokenizer.fit_on_texts(sentences)")

# Tokenizer
tokenizer = Tokenizer(
    lower = False,
    filters = ""
)
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

print("Preprocessing...")
input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Make padding
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

max_sequence_len_file = open("./pretrained/max_sequence_len.txt", "w", encoding="utf-8")
max_sequence_len_file.write(str(max_sequence_len))
max_sequence_len_file.close()

# Create data
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

print("Building model...")

if ch1 == 2:
    class TextGen(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, rnn_units):
            super(TextGen, self).__init__()
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.lstm1 = tf.keras.layers.LSTM(rnn_units)
            self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

        def call(self, inputs):
            x = self.embedding(inputs)
            x = self.lstm1(x)
            output = self.dense(x)
            return output

# EXPERIMENTAL: Model based on Transformer
if ch1 == 1:
    class TextGen(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim):
            super(TextGen, self).__init__()
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.transformer1 = Transformer(
                input_dim=embedding_dim,
                depth=4,
                sequence_len=max_sequence_len,
                causal=True,
                heads=4
            )
            self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
            self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

        def call(self, inputs):
            x = self.embedding(inputs)
            x = self.transformer1(x)
            x = self.global_average_pooling(x)
            output = self.dense(x)
            return output

if ch1 == 3:
    class TextGen(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, rnn_units):
            super(TextGen, self).__init__()
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)
            self.attention = tf.keras.layers.Attention()
            self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

        def call(self, inputs):
            x = self.embedding(inputs)
            lstm_output, _, _ = self.lstm(x)
            
            # Applying attention mechanism
            context_vector = self.attention([lstm_output, lstm_output])

            # Summing along the time axis to get context vector
            context_vector = tf.reduce_sum(context_vector, axis=1)

            output = self.dense(context_vector)
            return output

embedding_dim = 128

model = TextGen(
    vocab_size = total_words,
    embedding_dim = embedding_dim,
    rnn_units = 128
    #num_heads=3
    # filters=128,
    # kernel_size=2
    #max_seq_length=max_sequence_len
)

model.compile(
    loss = tf.keras.losses.CategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam( # not Adam
        learning_rate=0.003
    ),
    metrics = ['accuracy']
)

# TensorBoard
print("Loading TensorBoard...")
from tensorflow.keras.callbacks import TensorBoard
import datetime
log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

print("Training...")
# Training
batch_size = 128
model.fit(xs, ys, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback])

# Saving
model.save('./pretrained/model')

# Saving tokenizer
tokenizer_json = tokenizer.to_json()
with open('./pretrained/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))