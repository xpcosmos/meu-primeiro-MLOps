from flask import Flask
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('/Users/mikeiasoliveira/Documents/Projetos/meu-primeiro-MLOps/data/data.csv')
X = df.tamanho
y = df.preco
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= SEED, test_size=0.3)
model = LinearRegression()
model.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt', to= 'en')
    polaridade = tb_en.sentiment.polarity
    return f"Polaridade: {polaridade}"
    
app.run(debug=True)