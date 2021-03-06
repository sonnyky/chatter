import tensorflow as tf
import tensorflow_text
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

reloaded = tf.saved_model.load('chatter_engine')
user_chat = ''

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument("data")

class Chat(Resource):
    def post(self):
        args = parser.parse_args()

        print(type(args["data"]))
        print(args["data"])
        user_chat = tf.constant([args["data"]])
        result = reloaded.tf_generate_chat(input_text=user_chat)
        result_string = result['text'][0].numpy().decode()
        return result_string

api.add_resource(Chat, '/chat')
if __name__ == '__main__':
    app.run(debug=True)