import tensorflow as tf
import tensorflow_text as text
from utils.DataPreprocessing import load_data, tf_lower_and_split_punct

def rep_data_gen():
    question, answer = load_data('./input.txt', './output.txt')

    BUFFER_SIZE = len(question)
    BATCH_SIZE = 32

    dataset = tf.data.Dataset.from_tensor_slices((question, answer)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    for i in dataset:
        yield [i]

converter = tf.lite.TFLiteConverter.from_saved_model(
    saved_model_dir='chatter_engine',
    signature_keys=['serving_default']
)  # path to the SavedModel directory
print('after load')
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.representative_dataset = rep_data_gen

print('before conversion')
tflite_model = converter.convert()
print('after conversion')

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
print('after write')
