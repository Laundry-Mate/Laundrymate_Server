from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import csv
from data import washing_algorithm
import numpy as np
import json



app = Flask(__name__)
CORS(app)  # Add this line to enable CORS support


@app.route('/')
def hello():

  result = washing_algorithm()
  print(result)
  return 'hello'

@app.route('/double', methods=['POST', 'OPTIONS'])
def double_number():
  if request.method == 'OPTIONS':
      # Respond to the preflight request
      headers = {
          'Access-Control-Allow-Origin': 'http://localhost:5173',
          'Access-Control-Allow-Methods': 'POST',
          'Access-Control-Allow-Headers': 'Content-Type'
      }
      return '', 204, headers

  # Handle the actual POST request
  print(request);
  
  data = request.data.decode('utf-8')
  # print(data);
  # print(type(data));
  filename = 'clothes_dataset.csv'

  with open(filename, 'w') as csvfile:
    csvfile.write(data)
  
  result = washing_algorithm()
  print('result')
  print(result)

  return jsonify(result)

@app.route('/test')
def test():
  return 'returning from flask server'

def main():
  app.run(host='127.0.0.1', debug=False, port=80)

if __name__ == '__main__':
  main()