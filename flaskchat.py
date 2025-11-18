from flask import Flask, render_template, request, jsonify, session
import numpy as np
import google.generativeai as genai
import pandas as pd
import pickle
import joblib
from sklearn.tree import DecisionTreeClassifier
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'loan-pred-chatbot-secret-key-2024'

# Configure Gemini
genai.configure(api_key='AIzaSyDeMZy0c-tgxHdhdTBXp9h7CGQo3tVAq_Q')

def initialize_session_state():
    if "messages" not in session:
        session["messages"] = [
            {
                "role": "assistant", 
                "content": "Hello! I'm your loan eligibility assistant. Are you ready to begin? (Yes/No)",
                "timestamp": datetime.now().strftime("%H:%M")
            }
        ]
        session["started"] = False
        session["current_step"] = -1
        session["responses"] = {}
        session["show_next_question"] = True

def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_data(gender, married, dependents, education, employed, credit, area, 
                   ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
    try:
        male = 1 if gender.lower() == "male" else 0
        married_yes = 1 if married.lower() == "yes" else 0
        
        if dependents == '1':
            dependents_1, dependents_2, dependents_3 = 1, 0, 0
        elif dependents == '2':
            dependents_1, dependents_2, dependents_3 = 0, 1, 0
        elif dependents == "3+":
            dependents_1, dependents_2, dependents_3 = 0, 0, 1
        else:
            dependents_1, dependents_2, dependents_3 = 0, 0, 0

        not_graduate = 1 if education.lower() == "not graduate" else 0
        employed_yes = 1 if employed.lower() == "yes" else 0
        semiurban = 1 if area.lower() == "semiurban" else 0
        urban = 1 if area.lower() == "urban" else 0

        ApplicantIncomelog = np.log(float(ApplicantIncome))
        totalincomelog = np.log(float(ApplicantIncome) + float(CoapplicantIncome))
        LoanAmountlog = np.log(float(LoanAmount))
        Loan_Amount_Termlog = np.log(float(Loan_Amount_Term))
        
        if float(credit) <= 1000 and float(credit) >= 800:
            credit = 1
        else:
            credit = 0

        return [
            credit, ApplicantIncomelog, LoanAmountlog, Loan_Amount_Termlog, totalincomelog,
            male, married_yes, dependents_1, dependents_2, dependents_3, not_graduate, employed_yes, semiurban, urban
        ]
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

@app.route('/')
def index():
    initialize_session_state()
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    # Initialize session state if not exists
    initialize_session_state()
    
    # Define questions
    questions = [
        "What is your gender? (Male/Female)",
        "Are you married? (Yes/No)",
        "How many dependents do you have? (0/1/2/3+)",
        "What is your education level? (Graduate/Not Graduate)",
        "Are you self-employed? (Yes/No)",
        "What is your monthly applicant income?",
        "What is your monthly co-applicant income?",
        "What is the loan amount you are requesting?",
        "What is the loan term in days?",
        "What is your credit history score? (300-850)",
        "What is the property area? (Urban/Semiurban/Rural)"
    ]
    
    response_data = {
        "messages": [],
        "current_step": session["current_step"],
        "completed": False
    }
    
    # Handle initial state
    if not session["started"]:
        session["messages"].append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        if user_input.lower() == "yes":
            session["started"] = True
            session["current_step"] = 0
            session["messages"].append({
                "role": "assistant",
                "content": "Great! Let's get started with your loan eligibility assessment:\n\nWhat is your gender? (Male/Female)",
                "timestamp": datetime.now().strftime("%H:%M")
            })
        else:
            session["messages"].append({
                "role": "assistant",
                "content": "No problem! Let me know when you're ready to begin by typing 'Yes'.",
                "timestamp": datetime.now().strftime("%H:%M")
            })
    
    # Handle questionnaire responses
    elif session["current_step"] < len(questions):
        # Add user response to messages and store it
        session["messages"].append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        session["responses"][session["current_step"]] = user_input

        # Validate input
        valid_input = True
        error_message = ""

        # Input validation based on step
        if session["current_step"] in [5, 6, 7, 8]:  # Numeric inputs
            try:
                float(user_input)
            except ValueError:
                valid_input = False
                error_message = "Please enter a valid number."
        elif session["current_step"] == 9:  # Credit score
            try:
                score = float(user_input)
                if not (0 <= score <= 1000):
                    valid_input = False
                    error_message = "Credit score must be between 0 and 1000."
            except ValueError:
                valid_input = False
                error_message = "Please enter a valid credit score."

        if valid_input:
            session["current_step"] += 1
            if session["current_step"] < len(questions):
                session["messages"].append({
                    "role": "assistant",
                    "content": questions[session["current_step"]],
                    "timestamp": datetime.now().strftime("%H:%M")
                })
        else:
            session["messages"].append({
                "role": "assistant",
                "content": error_message,
                "timestamp": datetime.now().strftime("%H:%M")
            })
    
    # Process final results
    if session["started"] and session["current_step"] == len(questions):
        try:
            # Get all responses
            responses = session["responses"]
            
            # Extract all inputs
            gender = responses[0]
            married = responses[1]
            dependents = responses[2]
            education = responses[3]
            self_employed = responses[4]
            applicant_income = responses[5]
            coapplicant_income = responses[6]
            loan_amount = responses[7]
            loan_amount_term = responses[8]
            credit_history = responses[9]
            property_area = responses[10]

            # Display captured information
            captured_info = f"""Here is the information you provided:

• Gender: {gender}
• Marital Status: {married}
• Dependents: {dependents}
• Education: {education}
• Self-Employed: {self_employed}
• Applicant Income: ${applicant_income}
• Coapplicant Income: ${coapplicant_income}
• Loan Amount: ${loan_amount}
• Loan Term: {loan_amount_term} days
• Credit History: {credit_history}
• Property Area: {property_area}
            """
            
            session["messages"].append({
                "role": "assistant", 
                "content": captured_info,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            session["messages"].append({
                "role": "assistant", 
                "content": "We will now process your loan eligibility...",
                "timestamp": datetime.now().strftime("%H:%M")
            })

            # Construct the prompt for Gemini
            prompt = f"""
I want to check my eligibility for a loan. Here is my information:

Gender: {gender}
Marital Status: {married}
Dependents: {dependents}
Education: {education}
Self-Employed: {self_employed}
Applicant Income: {applicant_income}
Coapplicant Income: {coapplicant_income}
Loan Amount: {loan_amount}
Loan Amount Term: {loan_amount_term}
Credit History: {credit_history}
Property Area: {property_area}

Please evaluate the above details and provide a comprehensive analysis of my loan eligibility.
            """

            # Process with Gemini
            tools = genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name='predict_loan_status',
                        description="Predicts loan approval status based on user-provided details.",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "gender": genai.protos.Schema(type=genai.protos.Type.STRING),
                                "married": genai.protos.Schema(type=genai.protos.Type.STRING),
                                "dependents": genai.protos.Schema(type=genai.protos.Type.STRING),
                                "education": genai.protos.Schema(type=genai.protos.Type.STRING),
                                "self_employed": genai.protos.Schema(type=genai.protos.Type.STRING),
                                "applicant_income": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                "coapplicant_income": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                "loan_amount": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                "loan_amount_term": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                "credit_history": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                                "property_area": genai.protos.Schema(type=genai.protos.Type.STRING),
                            },
                            required=["gender", "married", "education", "applicant_income", "loan_amount", "credit_history", "property_area"]
                        )
                    )
                ]
            )

            try:
                # Create the model and start the chat
                model = genai.GenerativeModel(model_name='gemini-2.5-flash', tools=[tools])
                chat = model.start_chat(enable_automatic_function_calling=True)
                response = chat.send_message(prompt)

                # Parse function call arguments and make prediction
                # ... (your existing prediction logic here)

                # For demo purposes, let's simulate a response
                final_response = "Based on your information, I've analyzed your loan eligibility. Please check the result box below for detailed recommendations."

                session["messages"].append({
                    "role": "assistant", 
                    "content": final_response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                response_data["completed"] = True

            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                session["messages"].append({
                    "role": "assistant", 
                    "content": error_msg,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                print(error_msg)

        except Exception as e:
            error_msg = f"Error processing result: {str(e)}"
            session["messages"].append({
                "role": "assistant", 
                "content": error_msg,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            print(error_msg)
    
    # Update session
    session.modified = True
    
    # Prepare response
    response_data["messages"] = session["messages"]
    response_data["current_step"] = session["current_step"]
    
    return jsonify(response_data)

@app.route('/reset', methods=['POST'])
def reset_chat():
    session.clear()
    initialize_session_state()
    return jsonify({"status": "success", "message": "Chat reset successfully"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)