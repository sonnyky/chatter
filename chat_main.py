import tensorflow as tf
import tensorflow_text

reloaded = tf.saved_model.load('chatter_engine')
user_chat = ''

while user_chat != '終了':
    user_chat = tf.constant([input('あなたの発言：')])
    result = reloaded.tf_generate_chat(input_text=user_chat)
    print(result['text'][0].numpy().decode())
