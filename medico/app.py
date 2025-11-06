from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import re
import logging
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
index = None
qa_data = None

class MedicalDiagnosisEngine:
    """Professional medical diagnosis engine"""
    
    DISEASE_SYMPTOM_MATRIX = {
        "Abdominal aortic aneurysm": {
            "symptoms": ["abdominal pain", "back pain", "pulsating abdomen", "tenderness"],
            "severity": "high",
            "article": "Abdominal-aortic-aneurysm"
        },
        "Acute cholecystitis": {
            "symptoms": ["abdominal pain", "fever", "nausea", "vomiting", "jaundice", "right upper pain"],
            "severity": "high", 
            "article": "Acute-Cholecystitis"
        },
        "Acne": {
            "symptoms": ["pimples", "blackheads", "whiteheads", "red bumps", "face itching", "skin scarring"],
            "severity": "low",
            "article": "Acne"
        },
        "Anaphylaxis": {
            "symptoms": ["itching", "rash", "swelling", "difficulty breathing", "wheezing", "dizziness"],
            "severity": "emergency",
            "article": "Anaphylaxis"
        },
        "Acute lymphoblastic leukaemia": {
            "symptoms": ["fatigue", "pale skin", "frequent infections", "bruising", "bone pain", "bleeding"],
            "severity": "high",
            "article": "Acute-lymphoblastic-leukaemia"
        },
        "Bone cancer": {
            "symptoms": ["bone pain", "swelling", "fractures", "weight loss", "fatigue", "night sweats"],
            "severity": "high",
            "article": "Bone-cancer"
        },
        "Binge eating disorder": {
            "symptoms": ["binge eating", "eating quickly", "eating alone", "guilt after eating", "secret eating"],
            "severity": "medium",
            "article": "Binge-eating-disorder"
        },
        "Influenza": {
            "symptoms": ["fever", "cough", "sore throat", "runny nose", "muscle aches", "headache", "fatigue"],
            "severity": "medium",
            "article": "Influenza"
        },
        "Gastroenteritis": {
            "symptoms": ["diarrhea", "vomiting", "abdominal pain", "nausea", "fever", "dehydration"],
            "severity": "medium",
            "article": "Gastroenteritis"
        },
        "Migraine": {
            "symptoms": ["headache", "nausea", "sensitivity to light", "sensitivity to sound", "aura"],
            "severity": "medium",
            "article": "Migraine"
        }
    }
    
    @staticmethod
    def diagnose(symptoms_text):
        """Diagnose based on symptom patterns"""
        symptoms_text = symptoms_text.lower()
        matched_diseases = []
        
        for disease, info in MedicalDiagnosisEngine.DISEASE_SYMPTOM_MATRIX.items():
            symptom_matches = 0
            total_symptoms = len(info["symptoms"])
            
            for symptom in info["symptoms"]:
                if symptom in symptoms_text:
                    symptom_matches += 1
            
            # Calculate match percentage
            if symptom_matches > 0:
                match_percentage = (symptom_matches / total_symptoms) * 100
                if match_percentage >= 40:  # At least 40% symptom match
                    matched_diseases.append({
                        "disease": disease,
                        "match_percentage": match_percentage,
                        "severity": info["severity"],
                        "article": info["article"],
                        "matched_symptoms": symptom_matches,
                        "total_symptoms": total_symptoms
                    })
        
        # Sort by match percentage (highest first)
        matched_diseases.sort(key=lambda x: x["match_percentage"], reverse=True)
        return matched_diseases[:3]  # Return top 3 matches

def initialize_system():
    global model, index, qa_data
    
    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    try:
        # Load dataset
        info = pd.read_csv('bot.csv')
        info['Questions'] = info['Questions'].fillna('')
        
        # Validate dataset structure
        if 'Questions' not in info.columns or 'Answers' not in info.columns:
            raise ValueError("Invalid dataset structure")
            
        qa_data = info[['Questions', 'Answers']].to_dict('records')
        logger.info(f"Loaded {len(qa_data)} medical Q&A pairs")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Initialize with basic medical responses
        qa_data = [
            {'Questions': 'hello', 'Answers': 'Hello! Please describe your symptoms for medical diagnosis.'},
            {'Questions': 'symptoms', 'Answers': 'Please describe your symptoms in detail for accurate diagnosis.'}
        ]
    
    # Create embeddings for all questions
    questions = [item['Questions'] for item in qa_data]
    question_embeddings = model.encode(questions)
    
    # Create FAISS index
    dimension = question_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(question_embeddings)
    index.add(question_embeddings)
    
    save_system_state()

def save_system_state():
    """Save system state efficiently"""
    try:
        state = {
            'qa_data': qa_data,
            'index': index,
        }
        with open('bot_system_state.pkl', 'wb') as f:
            pickle.dump(state, f)
    except Exception as e:
        logger.error(f"Error saving system state: {e}")

def load_system_state():
    """Load system state"""
    global model, index, qa_data
    
    try:
        if os.path.exists('bot_system_state.pkl'):
            with open('bot_system_state.pkl', 'rb') as f:
                state = pickle.load(f)
            
            qa_data = state['qa_data']
            index = state['index']
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return True
    except Exception as e:
        logger.error(f"Error loading saved state: {e}")
    
    return False

def generate_diagnosis_response(matched_diseases, user_input):
    """Generate professional diagnosis response with article links"""
    if not matched_diseases:
        return "Based on your symptoms, I couldn't find a clear match in my medical database. Please consult a healthcare professional for accurate diagnosis."
    
    response = "🔍 **Medical Analysis Results:**\n\n"
    response += f"Based on your symptoms: *{user_input}*\n\n"
    response += "**Possible Conditions:**\n\n"
    
    for i, disease_info in enumerate(matched_diseases, 1):
        disease = disease_info["disease"]
        match_percentage = disease_info["match_percentage"]
        severity = disease_info["severity"]
        article = disease_info["article"]
        matched_symptoms = disease_info["matched_symptoms"]
        total_symptoms = disease_info["total_symptoms"]
        
        # Create article link
        article_url = f"http://127.0.0.1:5500/medico/Diseases/{article}.html"
        disease_link = f'<a href="{article_url}" target="_blank" class="disease-link">{disease}</a>'
        
        response += f"{i}. **{disease_link}**\n"
        response += f"   - Match Confidence: {match_percentage:.1f}%\n"
        response += f"   - Severity: {severity.upper()}\n"
        response += f"   - Symptoms Matched: {matched_symptoms}/{total_symptoms}\n\n"
    
    # Add severity warnings
    emergency_diseases = [d for d in matched_diseases if d["severity"] == "emergency"]
    high_severity_diseases = [d for d in matched_diseases if d["severity"] == "high"]
    
    if emergency_diseases:
        response += "🚨 **URGENT**: Some matched conditions require immediate medical attention!\n\n"
    elif high_severity_diseases:
        response += "⚠️ **Important**: Some conditions may require prompt medical evaluation.\n\n"
    
    response += "💡 **Next Steps**: Click on any condition above to read detailed information about symptoms, causes, and treatments."
    
    return response

def find_best_match(user_input, threshold=0.5):
    """Find best medical response match with diagnosis"""
    try:
        # First, try semantic matching with existing Q&A
        user_embedding = model.encode([user_input])
        faiss.normalize_L2(user_embedding)
        
        similarities, indices = index.search(user_embedding, k=3)
        
        # Return best match if above threshold
        if similarities[0][0] >= threshold:
            best_match = qa_data[indices[0][0]]['Answers']
            
            # If it's a HYPERLINK response, return it directly
            if isinstance(best_match, str) and best_match.startswith('=HYPERLINK'):
                return best_match
        
        # If no good semantic match, use diagnosis engine
        diagnosis_results = MedicalDiagnosisEngine.diagnose(user_input)
        
        if diagnosis_results:
            return generate_diagnosis_response(diagnosis_results, user_input)
        else:
            return "I've analyzed your symptoms but couldn't find a clear match. Please provide more specific symptoms or consult a healthcare provider for accurate diagnosis."
        
    except Exception as e:
        logger.error(f"Error in medical matching: {e}")
        return "I apologize, but I'm having trouble processing your medical query. Please try again or consult a healthcare professional."

def add_new_qa_pair(question, answer):
    """Add new medical Q&A pair"""
    global qa_data, index
    
    try:
        # Add to data
        qa_data.append({'Questions': question, 'Answers': answer})
        
        # Encode and add to index
        new_embedding = model.encode([question])
        faiss.normalize_L2(new_embedding)
        index.add(new_embedding)
        
        # Save state
        save_system_state()
        
        # Update CSV
        new_data = pd.DataFrame({'Questions': [question], 'Answers': [answer]})
        new_data.to_csv('bot.csv', mode='a', header=False, index=False)
        
        return True, "Medical case added successfully."
        
    except Exception as e:
        logger.error(f"Error adding medical case: {e}")
        return False, "Error adding medical information."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.form['user_input'].strip()
        
        if not user_input:
            return jsonify({
                'response': 'Please describe your symptoms for medical diagnosis.',
                'is_medical': True
            })
        
        # Get medical analysis
        medical_response = find_best_match(user_input)
        
        # Handle HYPERLINK responses
        if isinstance(medical_response, str) and medical_response.startswith('=HYPERLINK'):
            match = re.match(r'=HYPERLINK\("([^"]+)"\,"([^"]+)"\)', medical_response)
            if match:
                url, display_text = match.groups()
                medical_response = f'<a href="{url}" target="_blank" class="disease-link">{display_text}</a>'
                medical_response += '<br><br>💡 Click the link above to read detailed information about this condition.'
        
        return jsonify({
            'response': medical_response,
            'is_medical': True
        })
        
    except Exception as e:
        logger.error(f"Error in medical prediction: {e}")
        return jsonify({
            'response': 'I apologize for the technical difficulty. Please consult a healthcare provider for medical concerns.',
            'is_medical': True
        })

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle medical feedback"""
    try:
        data = request.json
        user_input = data.get('user_input', '')
        provided_answer = data.get('provided_answer', '')
        was_correct = data.get('was_correct', False)
        
        if not was_correct and provided_answer and len(provided_answer) > 5:
            success, message = add_new_qa_pair(user_input, provided_answer)
            return jsonify({
                'status': 'success' if success else 'error',
                'message': message
            })
        
        return jsonify({
            'status': 'success', 
            'message': 'Thank you for your medical feedback!'
        })
        
    except Exception as e:
        logger.error(f"Error in medical feedback: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Error processing medical feedback'
        })

@app.route('/admin/stats')
def system_stats():
    """System statistics"""
    return jsonify({
        'total_medical_cases': len(qa_data),
        'diagnosis_engine_diseases': len(MedicalDiagnosisEngine.DISEASE_SYMPTOM_MATRIX),
        'index_size': index.ntotal if index else 0,
        'system_status': 'operational'
    })

if __name__ == '__main__':
    # Initialize system
    if not load_system_state():
        logger.info("Initializing medical diagnosis system...")
        initialize_system()
    else:
        logger.info("Medical system state loaded successfully")
    
    app.run(debug=True, port=7000)