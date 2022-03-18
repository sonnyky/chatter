import tensorflow as tf
import tensorflow_text
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import os

app = Flask(__name__)
api = Api(app)

APP_ROOT = os.getcwd()
app.config['MODEL'] = os.path.join(APP_ROOT, 'model')

reloaded = tf.saved_model.load(os.path.join(app.config['MODEL'], 'chatter_engine'))
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

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class status (Resource):
    def get(self):
        try:
            return {'data': 'Api is Running'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}

api.add_resource(HelloWorld, '/')

api.add_resource(Chat, '/chat')
if __name__ == '__main__':
    app.run()