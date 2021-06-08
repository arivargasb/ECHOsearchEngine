from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import nmslib
from gensim.models.fasttext import FastText

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/search',methods=['POST'])
def search():

	# load data
	df = pd.read_csv ('data/echo_test.csv', usecols= ['text','date'])

	ft_model = FastText(
    sg=1, # use skip-gram: usually gives better results
    size=100, # embedding dimension (default)
    window=10, # window size: 10 tokens before and 10 tokens after to get wider context
    min_count=5, # only consider tokens with at least n occurrences in the corpus
    negative=15, # negative subsampling: bigger than default to sample negative examples more
    min_n=2, # min character n-gram
    max_n=5) # max character n-gram

	# load document vectors
	with open( "notebooks/weighted_doc_vects_ECHO2.p", "rb" ) as f:
		weighted_doc_vects = pickle.load(f)

	# create a random matrix to index
	data = np.vstack(weighted_doc_vects)

	# initialize a new index, using a HNSW index on Cosine Similarity - can take a couple of mins
	index = nmslib.init(method='hnsw', space='cosinesimil')
	index.addDataPointBatch(data)
	index.createIndex({'post': 2}, print_progress=True)

	input = 'hydroxychloroquine'.lower().split()


	query = [ft_model[vec] for vec in input]
	query = np.mean(query,axis=0)

	ids, distances = index.knnQuery(query, k=5)

	if request.method == 'POST':
		for i,j in zip(ids,distances):
			my_search = df.date.values
		return render_template('result.html', retrieval1 = my_search[0],
		retrieval2 = my_search[1], retrieval3 = my_search[2], retrieval4 = my_search[3],
		retrieval5 = my_search[4])


if __name__ == '__main__':
	app.run(debug=True)