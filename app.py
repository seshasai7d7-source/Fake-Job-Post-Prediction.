import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle


 


app = Flask(__name__) #Initialize the flask App
app.secret_key = 'dev'  # required for `flash()` in preview route


model = None
vecs = None
classifiers = None
try:
	model = pickle.load(open('random.pickle', 'rb'))
except Exception as e:
	print(f"Warning: could not load random.pickle: {e}")
try:
	vecs = pickle.load(open('vectorizers.pickle', 'rb'))
except Exception as e:
	print(f"Warning: could not load vectorizers.pickle: {e}")
try:
	classifiers = pickle.load(open('classifiers.pickle', 'rb'))
except Exception as e:
	print(f"Warning: could not load classifiers.pickle: {e}")
 


@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/chart')
def chart():
	return render_template('chart.html')

@app.route('/performance')
def performance():
	return render_template('performance.html')
   

@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
	if request.method == 'POST':
		dataset = request.files.get('datasetfile')
		if dataset is None:
			flash('No file uploaded')
			return redirect(request.url)
		try:
			df = pd.read_csv(dataset, encoding='unicode_escape')
		except Exception as e:
			flash(f'Could not read CSV: {e}')
			return redirect(request.url)
		if 'Id' in df.columns:
			df.set_index('Id', inplace=True)
		return render_template("preview.html", df_view=df)
        
@app.route('/fake_prediction')
def fake_prediction():
    return render_template('fake_prediction.html')

  


@app.route('/predict',methods=['POST'])
def predict():
	if model is None:
		return render_template('fake_prediction.html', prediction_texts='Model not available. See server log.')
	try:
		features = [float(x) for x in request.form.values()]
	except Exception:
		return render_template('fake_prediction.html', prediction_texts='Invalid input features.')
	final_features = np.array([features])
	try:
		y_pred = model.predict(final_features)
	except Exception as e:
		return render_template('fake_prediction.html', prediction_texts=f'Prediction error: {e}')
	label = 'Unknown'
	if y_pred[0] == 1:
		label = "Fake Job Post"
	elif y_pred[0] == 0:
		label = "Legit Job Post"
	return render_template('fake_prediction.html', prediction_texts=label)
@app.route('/text_prediction')
def text_prediction():
    return render_template("text_prediction.html")

 

@app.route('/job')
def job():	
	abc = request.args.get('news', '')
	if not abc:
		return render_template('text_prediction.html', prediction_text='No input provided.')
	input_data = [abc.rstrip()]
	# transforming input
	if vecs is None or classifiers is None:
		return render_template('text_prediction.html', prediction_text='Text model not available. See server log.')
	# transforming input
	tfidf_test = vecs.transform(input_data)
	# predicting the input
	y_preds = classifiers.predict(tfidf_test)
	if y_preds[0] == 1:
		labels="Fake Job Post"
	elif y_preds[0] == 0:
		labels="Legit Job Post"
	return render_template('text_prediction.html', prediction_text=labels) 
    
    
if __name__ == "__main__":
    app.run(debug=True)
