from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import base64
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

#route to handle the csv file upload and predicion
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['csv_file']

    csv_content = file.read().decode('utf-8')

    df = pd.read_csv(io.StringIO(csv_content))

    #separate the claim_id column if it exists
    if 'claim_id' in df.columns:
        claim_ids = df['claim_id'].tolist()
        df = df.drop(columns=['claim_id'])
    else:
        claim_ids = None
    
    #send the data to the BentoML service for prediction
    try:
        response = requests.post('http://localhost:3000/predict',
                                    json={"data": df.to_dict(orient='records')})
        response.raise_for_status()  # Raise error for bad status codes
        result = response.json()
        predictions = result['predictions']
    except requests.exceptions.RequestException as e:
        return f"Error connecting to ML service: {str(e)}"
    except KeyError:
        return f"Unexpected response from ML service: {result}"
    
    df['Prediction'] = predictions

    #Reattach claim_id if it was separated
    if claim_ids is not None:
        df['claim_id'] = claim_ids

    #reorder columns to have claim_id first if it exists
    if 'claim_id' in df.columns:
        df = df[['claim_id'] + [col for col in df.columns if col != 'claim_id']]
    
    #render the results in the html template
    return render_template('results.html', tables=[df.to_html(classes='data', header="true")])

if __name__ == '__main__':
    app.run(debug=True, port=5005)
