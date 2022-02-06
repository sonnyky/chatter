# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf
import tensorflow_text as tf_text
from utils.DataPreprocessing import load_data, tf_lower_and_split_punct

import model.TrainChatter as TrainChatter
import model.MaskedLoss as MaskedLoss
import model.BatchLogs as BatchLogs

import chat_generator.ChatGenerator as ChatGenerator

# Press the green button in the gutter to run the script.
print('Starting')
question, answer = load_data('./input.txt', './output.txt')

BUFFER_SIZE = len(question)
BATCH_SIZE = 32

dataset = tf.data.Dataset.from_tensor_slices((question, answer)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)

max_vocab_size = 5000

input_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)
input_text_processor.adapt(question)

# Here are the first 10 words from the vocabulary:
print(input_text_processor.get_vocabulary()[:10])

output_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

output_text_processor.adapt(answer)
output_text_processor.get_vocabulary()[:10]

embedding_dim = 256
units = 1024
train_chatter = TrainChatter.TrainChatter(embedding_dim, units, input_text_processor=input_text_processor, output_text_processor=output_text_processor)
train_chatter.compile(optimizer=tf.optimizers.Adam(), loss=MaskedLoss.MaskedLoss())

batch_loss = BatchLogs.BatchLogs('batch_loss')
train_chatter.fit(dataset, epochs=1,
                     callbacks=[batch_loss])

chatter_engine = ChatGenerator.ChatGenerator(
    encoder=train_chatter.encoder,
    decoder=train_chatter.decoder,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
)

input_text = tf.constant([
    'こんにちは', # "It's really cold here."
    '寒いね', # "This is my life.""
])

result = chatter_engine.tf_generate_chat(
    input_text = input_text)

print(result['text'][0].numpy().decode())
print(result['text'][1].numpy().decode())
print()

## Save model
print('Saving model')
tf.saved_model.save(chatter_engine, 'chatter_engine',
                    signatures={'serving_default': chatter_engine.tf_generate_chat})