
import  pandas as pd
from    flask import Flask, request, jsonify
from    textblob import TextBlob
from    sklearn.linear_model import LinearRegression
from    sklearn.model_selection import train_test_split


SEED = 42

df = pd.read_csv('./data/data.csv')

columns = ['tamanho', 'ano', 'garagem']

X = df.drop('preco', axis=1)
y = df.preco

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= SEED, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)

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