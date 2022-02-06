import pathlib
import tensorflow_text as tf_text
import tensorflow as tf

def inint_preprocessing(input):
    pass

def load_data(path_to_question_data, path_to_answer_data):
    q_path = pathlib.Path(path_to_question_data)
    a_path = pathlib.Path(path_to_answer_data)
    question = q_path.read_text(encoding='utf-8')
    answer = a_path.read_text(encoding='utf-8')
    q_lines = question.splitlines()
    a_lines = answer.splitlines()
    return q_lines, a_lines

def tf_lower_and_split_punct(text):
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text