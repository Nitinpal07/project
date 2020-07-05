#Importing the Libraries
import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
import os
import sklearn
import joblib
import pickle
import flask
import os
import newspaper
from newspaper import Article
import urllib
import json

#Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open('model.pickle', 'rb') as handle:
	model = pickle.load(handle)

@app.route('/')
def main():
    return render_template('index.html')

#Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/search_by_url',methods=['POST'])
def predict():
    result = request.form
    print(result)
    url = result['url']
    print(url)
    # url =request.get_data(as_text=True)[5:]
    # url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    print(news)
    # #Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    #return render_template('main.html', prediction_text='The news is "{}"'.format(pred[0]))
    return f'<html><body><h1>{pred[0]}</h1> <form action="/"> <button type="submit">back </button> </form></body></html>'

@app.route('/search_by_text',methods=['POST'])
def get_text():

    result=request.form
    query_title = result['title']
    query_author = result['author']
    query_text = result['maintext']
    print(query_text)
    query = query_text + query_author +query_text
    pred = model.predict([query])
    print(pred)
    return f'<html><body><h1>{pred[0]}</h1> <form action="/"> <button type="submit">back </button> </form></body></html>'
    
if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)