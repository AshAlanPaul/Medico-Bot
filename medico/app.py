from flask import Flask, render_template, request, send_from_directory, url_for, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import os
import re

app = Flask(__name__)

# Define ml and vectorizer as global variables
ml = None
vectorizer = None
 
def train_model(X, y):
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    ml = DecisionTreeClassifier()
    ml.fit(X_vectorized, y)
    return ml, vectorizer
          
def get_user_feedback(user_input, model, vectorizer):
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vectorized)
    return prediction[0]
           
def update_model(user_input, correct_answer, X, y, model, vectorizer):
    updated_data = pd.DataFrame({'Questions': [user_input], 'Answers': [correct_answer]})
    updated_df = pd.concat([X, updated_data], ignore_index=True)
    updated_df = updated_df.dropna(subset=['Questions'])
    updated_X = updated_df['Questions']
    updated_y = updated_df['Answers']
    model, vectorizer = train_model(updated_X, updated_y)
    return model, vectorizer, updated_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global ml, vectorizer  # Declare ml and vectorizer as global
    user_input = request.form['user_input']
    model_response = get_user_feedback(user_input, ml, vectorizer)

    # Generate the URL for serving images dynamically
    user_image_url = url_for('serve_images', filename='user.png')
    bot_image_url = url_for('serve_images', filename='bot.png')

    # Check if the model_response starts with '=HYPERLINK'
    if model_response.startswith('=HYPERLINK'):
        # Extract the URL and display text from the HYPERLINK formula
        match = re.match(r'=HYPERLINK\("([^"]+)"\,"([^"]+)"\)', model_response)
        if match:
            url, display_text = match.groups()
            # Create a hyperlink
            model_response = f'<a href="{url}">{display_text}</a>'
    else:
        # If it's not a hyperlink, treat it as plain text
        model_response = model_response

    # Return a JSON response containing the updated model response and image URLs
    return jsonify({'response': model_response, 'user_image_url': user_image_url, 'bot_image_url': bot_image_url})
    
@app.route('/images/<path:filename>') 
def serve_images(filename):
    return send_from_directory(os.path.join(app.root_path, 'images'), filename)

if __name__ == '__main__':
    info = pd.read_csv('bot.csv')
    info['Questions'] = info['Questions'].fillna('')
    X = info['Questions']
    y = info['Answers']
    ml, vectorizer = train_model(X, y)
    app.run(debug=True, port=7000)
              