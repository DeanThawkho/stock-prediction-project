import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template
from model.final import main

app = Flask(__name__, template_folder='../templates', static_folder='../static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data['symbol'].upper()
        days = int(data['days'])

        predictions, plot = main(symbol, days)
        return jsonify({
            'predicted_price': float(predictions[-1]),
            'plot' : plot
        })
    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host = '127.0.0.1', port = 5000, debug = True, use_reloader = False)