
import  pandas as pd
from    flask import Flask, request, jsonify
from    textblob import TextBlob
from    sklearn.linear_model import LinearRegression

import pickle


SEED = 42
columns = ['tamanho', 'ano', 'garagem']

model = pickle.load(open('model.sav', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/cotacao/', methods=['POST'])
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in columns]
    
    preco = model.predict([dados_input])
    return jsonify(preco=preco[0])


@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt', to= 'en')
    polaridade = tb_en.sentiment.polarity
    return f"Polaridade: {polaridade}"

    
app.run(debug=True)