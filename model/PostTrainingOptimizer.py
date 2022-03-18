import tensorflow as tf

class PostTrainingOptimizer():
    def convert_saved_model(self, path_to_model):
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_saved_model(path_to_model)  # path to the SavedModel directory
        tflite_model = converter.convert()

        # Save the model.
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)