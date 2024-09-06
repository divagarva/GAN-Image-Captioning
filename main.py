import nltk
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3 as inception
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Input, add
from tensorflow.keras.models import Model, model_from_json
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

# Download and set up NLTK tokenizer
nltk.data.path = ['/Users/divagarvakeesan/nltk_data']  # Adjust this path as needed
nltk.download('punkt', download_dir='/Users/divagarvakeesan/nltk_data')

# Load the InceptionV3 model with pre-trained weights
encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)

# Constants
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
START = "startseq"
STOP = "endseq"
vocab_size = None
max_length = None
embedding_dim = 300
EPOCHS = 10
number_pics_per_batch = 6

# Function to preprocess and encode an image
preprocess_input = inception.preprocess_input


def encodeImage(img):
    img = img.resize((WIDTH, HEIGHT))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = encode_model.predict(x)  # Get the encoding vector for the image
    x = np.reshape(x, OUTPUT_DIM)
    return x


# Load your dataset CSV
csv_path = os.path.join(os.getcwd(), "image_caption_map.csv")  # Adjust the path if needed
data = pd.read_csv(csv_path)

# Process captions and encode the images
data['caption'] = data['caption'].apply(lambda x: START + ' ' + x + ' ' + STOP)
encoded_images = {}
remove_these = []

# Encode images
for i in range(data.shape[0]):
    image_file_name = data['file_name'][i]
    image_path = os.path.join(os.getcwd(), 'images', image_file_name)
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(HEIGHT, WIDTH))
        encoded_images[int(image_file_name.split('.')[0])] = encodeImage(img)
    except Exception as e:
        print(f"Error processing image {image_file_name}: {e}")
        remove_these.append(image_file_name)

data = data[~data['file_name'].isin(remove_these)].reset_index(drop=True)
data['id'] = data['file_name'].apply(lambda x: int(x.split('.')[0]))

# Clean up captions
punctuation = re.escape("!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~")
data['caption'] = data['caption'].apply(lambda x: re.sub(f"[{punctuation}]", ' ', x))
data['caption'] = data['caption'].apply(lambda x: re.sub("\d", ' ', x))
data['caption'] = data['caption'].apply(lambda x: re.sub("\s+", ' ', x))
data['caption'] = data['caption'].str.lower()


# Tokenize captions and create vocab
def tokenize_caption(caption):
    return caption.split()


word_count_threshold = 5
word_counts = {}
for caption in data['caption']:
    tokens = tokenize_caption(caption)
    for word in tokens:
        word_counts[word] = word_counts.get(word, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
idxtoword = {}
wordtoidx = {}
ix = 1
for w in vocab:
    wordtoidx[w] = ix
    idxtoword[ix] = w
    ix += 1

vocab_size = len(idxtoword) + 1
max_length = max([len(tokenize_caption(caption)) for caption in data['caption']])

# Load GloVe embeddings
glove_path = os.path.join(os.getcwd(), "glove.42B.300d.txt")
embeddings_index = {}

with open(glove_path, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoidx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Data Generator
def data_generator(data, encoded_images, wordtoidx, max_length, num_photos_per_batch):
    x1, x2, y = [], [], []
    n = 0
    while True:
        for k, caption in enumerate(data['caption']):
            n += 1
            photo = encoded_images[data['id'][k]]
            seq = [wordtoidx[word] for word in tokenize_caption(caption) if word in wordtoidx]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                x1.append(photo)
                x2.append(in_seq)
                y.append(out_seq)
            if n == num_photos_per_batch:
                x1 = tf.convert_to_tensor(np.array(x1), dtype=tf.float32)
                x2 = tf.convert_to_tensor(np.array(x2), dtype=tf.int32)
                y = tf.convert_to_tensor(np.array(y), dtype=tf.float32)
                yield ((x1, x2), y)
                x1, x2, y = [], [], []
                n = 0


# Define the output signature
output_signature = (
    (
        tf.TensorSpec(shape=(None, OUTPUT_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)
    ),
    tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
)


# Create the dataset from the generator
def dataset_generator():
    return data_generator(data, encoded_images, wordtoidx, max_length, number_pics_per_batch)


dataset = tf.data.Dataset.from_generator(
    dataset_generator,
    output_signature=output_signature
)

# Model Definition
inputs1 = Input(shape=(OUTPUT_DIM,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# Load GloVe embeddings into the embedding layer
caption_model.layers[2].set_weights([embedding_matrix])
caption_model.layers[2].trainable = False
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
steps = len(data['caption']) // number_pics_per_batch
for i in range(EPOCHS):
    caption_model.fit(dataset, epochs=1, steps_per_epoch=steps, verbose=1)

# Save the model weights with the correct extension
caption_model.save_weights('caption_model.weights.h5')


# Caption generation function with image display
def generate_and_display_caption(photo, image_path):
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)

    # Display the image with the generated caption
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(final_caption)
    plt.show()


# Test the model with image display on multiple test images
def test_multiple_images(indices):
    for index in indices:
        image_file = data.iloc[index, 2]
        image_path = os.path.join(os.getcwd(), 'images', image_file)
        image = encoded_images[int(image_file.split('.')[0])]
        image = image.reshape((1, OUTPUT_DIM))
        generate_and_display_caption(image, image_path)


# Test the model with a few images (adjust the indices as needed)
test_multiple_images([5, 287, 185, 549])

# Adjust learning rate and batch size for extended training
caption_model.optimizer.lr = 1e-4
number_pics_per_batch = 10
steps = len(data['caption']) // number_pics_per_batch

# Extended training with tqdm progress bar
for i in tqdm(range(EPOCHS)):
    generator = data_generator(data, encoded_images, wordtoidx, max_length, number_pics_per_batch)
    caption_model.fit(generator, epochs=30, steps_per_epoch=steps, verbose=1)

# Additional test case for generating and displaying the caption
def generate_and_display_multiple_captions(indices):
    for index in indices:
        try:
            image_file = data.iloc[index, 2]
            image = encoded_images[int(image_file.split('.')[0])]
            image = image.reshape((1, OUTPUT_DIM))
            image_path = os.path.join('images', image_file)

            # Generate and print the caption
            caption = generate_and_display_caption(image, image_path)
        except KeyError:
            print(f"Image with index {index} not found in the dataset.")


# Test case for generating and displaying captions for multiple images
indices_to_test = [5, 287, 185, 549, 79, 1500]  # Adjust these indices as needed
generate_and_display_multiple_captions(indices_to_test)


# Save model architecture and weights
model_json = caption_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
caption_model.save_weights("model.weights.h5")

# Loading the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.weights.h5")
loaded_model.summary()
