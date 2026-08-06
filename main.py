import streamlit as st
import json
import re
import numpy as np
import pandas as pd
from groq import Groq
from sklearn.ensemble import HistGradientBoostingClassifier

# ==============================================================================
# 1. PAGE SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="ABC Credit - Vehicle Loan Assistant",
    page_icon="🤖",
    layout="wide"
)

# Constants & Thresholds
EARLY_APPROVE_THRESH = 0.15
EARLY_DECLINE_THRESH = 0.80
FINAL_DECISION_THRESH = 0.44

MASTER_DATA = {
    "Employment_Type": ["SAL", "SEP", "STU", "AGR", "PEN", "NONEARNMEM", "NREGI", "NPP"],
    "Product_Code": ["MC", "SC", "MO", "EB"],
    "Make_Code": ["JUPITER", "RAIDER", "APACHE", "NTORQ", "RADEON", "RONIN", "MOPEDS", "SPORT", "ZEST", "TVS", "CITY PLUS"]
}

PREMIUM_WORDS = {
    "4V": 2, "310": 3, "350": 3, "200": 1, "165 RP": 2,
    "2CH": 2, "1CH": 1, "ABS": 2, "DISC": 1,
    "R MODE": 2, "RACE": 2, "RACE XP": 2, "SUPER SQUAD": 2,
    "SPL EDITION": 2, "WINNER EDITION": 1, "MATTE SERIES": 1,
    "CLASSIC": 1, "XT": 1, "ST": 2, "ALLOY": 1, "BSVI": 1,
    "OBDIIA": 1, "EV": 3, "IQUBE": 3
}

# ==============================================================================
# 2. MOCK MODEL SETUP (Cached for fast rendering)
# ==============================================================================
@st.cache_resource
def load_underwriting_model():
    """Generates a dummy HistGradientBoostingClassifier model for demonstration."""
    # Define expected columns
    cols = [
        'Loan_Amount', 'Vehicle_Price', 'LTV', 'Loan_to_Salary', 
        'Log_Loan_Amount', 'Premium_Score', 'Age', 'Monthly_Income', 
        'Has_Active_Loans', 'Employment_Type_SAL', 'Employment_Type_SEP', 
        'Product_Code_MC', 'Product_Code_SC', 'Make_Code_APACHE', 'Make_Code_TVS'
    ]
    # Synthetic training data
    X_dummy = pd.DataFrame(np.random.rand(100, len(cols)), columns=cols)
    y_dummy = np.random.choice([0, 1], size=100, p=[0.7, 0.3]) # 30% default risk
    
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_dummy, y_dummy)
    return model, cols

ml_model, MODEL_COLUMNS = load_underwriting_model()

# ==============================================================================
# 3. HELPER FUNCTIONS & LLM LOGIC
# ==============================================================================
def calculate_premium_score(model_desc: str) -> int:
    if not model_desc:
        return 0
    text = str(model_desc).upper()
    score = sum(weight for keyword, weight in PREMIUM_WORDS.items() if keyword in text)
    cc_match = re.search(r'(\d{2,4})\s*CC', text)
    if cc_match:
        cc_val = int(cc_match.group(1))
        if cc_val >= 650: score += 4
        elif cc_val >= 350: score += 2
        elif cc_val >= 200: score += 1
    return score

def build_model_features(p1_data: dict, p2_data: dict = None) -> pd.DataFrame:
    p2_data = p2_data or {}
    req_loan = float(p1_data.get('requested_loan_amount', 0))
    price = float(p1_data.get('vehicle_price', 1))
    salary = float(p2_data.get('monthly_income', 50000))
    
    ltv = (req_loan / price * 100.0) if price > 0 else 100.0
    loan_to_salary = req_loan / (salary * 12 + 1.0)
    prem_score = calculate_premium_score(p1_data.get('model_description', ''))
    
    payload = {
        'Loan_Amount': req_loan,
        'Vehicle_Price': price,
        'LTV': ltv,
        'Loan_to_Salary': loan_to_salary,
        'Log_Loan_Amount': np.log1p(req_loan),
        'Premium_Score': prem_score,
        'Age': int(p2_data.get('age', 30)),
        'Monthly_Income': salary,
        'Has_Active_Loans': int(p2_data.get('has_active_loans', 0)),
        'Employment_Type': p2_data.get('Employment_Type', 'SAL'),
        'Product_Code': p1_data.get('Product_Code', 'MC'),
        'Make_Code': p1_data.get('Make_Code', 'TVS')
    }
    return pd.get_dummies(pd.DataFrame([payload]))

def predict_risk(model, df_features: pd.DataFrame, model_columns: list) -> float:
    aligned_df = df_features.reindex(columns=model_columns, fill_value=0).astype(np.float32)
    return float(model.predict_proba(aligned_df)[0][1])

def extract_phase1_slots(client: Groq, user_input: str) -> dict:
    system_prompt = f"""
    You are an extraction engine for a two-wheeler loan system.
    Extract vehicle details and financial details from user input.
    Make_Code valid choices: {MASTER_DATA['Make_Code']}
    Product_Code choices: MC (Motorcycle), SC (Scooter), MO (Moped), EB (EV)
    
    Return ONLY JSON:
    {{
      "requested_loan_amount": FLOAT_OR_NULL,
      "vehicle_price": FLOAT_OR_NULL,
      "Product_Code": "ENUM_OR_NULL",
      "Make_Code": "ENUM_OR_NULL",
      "model_description": "STRING_OR_NULL",
      "model_variant": "STRING_OR_NULL"
    }}
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

def extract_phase2_slots(client: Groq, user_input: str) -> dict:
    system_prompt = """
    Extract customer demographics and map job to Employment_Type.
    Valid Employment_Type codes: SAL, SEP, STU, AGR, PEN, NONEARNMEM, NREGI, NPP.
    
    Return ONLY JSON:
    {
      "Employment_Type": "ENUM_OR_NULL",
      "monthly_income": FLOAT_OR_NULL,
      "age": INT_OR_NULL,
      "pincode": "STRING_OR_NULL",
      "has_active_loans": INT_0_OR_1
    }
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

def calculate_max_eligible_loan(model, p1_data: dict, p2_data: dict) -> dict:
    req_loan = p1_data['requested_loan_amount']
    vehicle_price = p1_data['vehicle_price']
    low, high, max_approved = 0.0, req_loan, 0.0
    
    for _ in range(12):
        mid = (low + high) / 2.0
        test_p1 = p1_data.copy()
        test_p1['requested_loan_amount'] = mid
        df_feat = build_model_features(test_p1, p2_data)
        risk = predict_risk(model, df_feat, MODEL_COLUMNS)
        
        if risk <= FINAL_DECISION_THRESH:
            max_approved = mid
            low = mid
        else:
            high = mid
            
    return {
        "requested_loan": req_loan,
        "max_eligible_loan": max_approved,
        "is_partial_possible": max_approved >= (0.30 * vehicle_price)
    }

# ==============================================================================
# 4. CHATBOT ENGINE ORCHESTRATOR
# ==============================================================================
def process_chat_message(client: Groq, user_input: str):
    state = st.session_state.chatbot_state
    step = state.get("step", "PHASE_1_COLLECTION")
    
    if step == "PHASE_1_COLLECTION":
        extracted = extract_phase1_slots(client, user_input)
        for k, v in extracted.items():
            if v is not None:
                state["p1_data"][k] = v
                
        p1_data = state["p1_data"]
        # Slot validation check
        missing = []
        if not p1_data.get('requested_loan_amount'): missing.append('requested_loan_amount')
        if not p1_data.get('vehicle_price'): missing.append('vehicle_price')
        
        if missing:
            if 'vehicle_price' in missing and p1_data.get('requested_loan_amount'):
                bot_msg = f"Got it, a loan request of ₹{p1_data['requested_loan_amount']:,.0f}! What is the vehicle model name and its expected on-road price?"
            else:
                bot_msg = "Please specify both the requested loan amount and the on-road vehicle price to proceed."
            return bot_msg
        
        # Fast Screening Check
        p1_features = build_model_features(p1_data)
        p1_risk = predict_risk(ml_model, p1_features, MODEL_COLUMNS)
        
        if p1_risk <= EARLY_APPROVE_THRESH:
            state["step"] = "COMPLETED"
            return f"🎉 Excellent news! Your loan request of ₹{p1_data['requested_loan_amount']:,.0f} for the {p1_data.get('model_description', 'vehicle')} is **INSTANTLY PRE-APPROVED**!"
        elif p1_risk >= EARLY_DECLINE_THRESH:
            offer = calculate_max_eligible_loan(ml_model, p1_data, {})
            state["step"] = "COMPLETED"
            return f"Thank you for applying. We cannot approve ₹{p1_data['requested_loan_amount']:,.0f}, but you are pre-approved for up to **₹{offer['max_eligible_loan']:,.0f}**."
        else:
            state["step"] = "PHASE_2_COLLECTION"
            return "Vehicle details noted! To complete full underwriting, what is your current occupation, monthly salary, age, and address pincode?"

    elif step == "PHASE_2_COLLECTION":
        extracted = extract_phase2_slots(client, user_input)
        for k, v in extracted.items():
            if v is not None:
                state["p2_data"][k] = v
                
        full_features = build_model_features(state["p1_data"], state["p2_data"])
        final_risk = predict_risk(ml_model, full_features, MODEL_COLUMNS)
        state["step"] = "COMPLETED"
        
        if final_risk < FINAL_DECISION_THRESH:
            req = state["p1_data"]["requested_loan_amount"]
            return f"🎉 Congratulations! Your loan request of **₹{req:,.0f}** has been **FULLY APPROVED**."
        else:
            offer = calculate_max_eligible_loan(ml_model, state["p1_data"], state["p2_data"])
            req = offer["requested_loan"]
            max_l = offer["max_eligible_loan"]
            if offer["is_partial_possible"]:
                return f"Thank you for applying. While we cannot approve ₹{req:,.0f}, based on your profile you are pre-approved for up to **₹{max_l:,.0f}**. Update your loan request to ₹{max_l:,.0f} to proceed."
            else:
                return f"Thank you for applying. We are unable to approve your loan request of ₹{req:,.0f} at this time. You may consider applying with a co-applicant."

    return "Session complete. Click 'Reset Application' in the sidebar to start over."

# ==============================================================================
# 5. UI INITIALIZATION & SIDEBAR
# ==============================================================================
st.title("🚗 ABC Credit - Intelligent Loan Assistant")

# Sidebar
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key to run Llama-3 extraction.")

if st.sidebar.button("🔄 Reset Application"):
    st.session_state.messages = []
    st.session_state.chatbot_state = {"step": "PHASE_1_COLLECTION", "p1_data": {}, "p2_data": {}}
    st.rerun()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome to ABC Credit! Which vehicle are you looking to buy, what is its on-road price, and how much loan do you need?"}
    ]

if "chatbot_state" not in st.session_state:
    st.session_state.chatbot_state = {"step": "PHASE_1_COLLECTION", "p1_data": {}, "p2_data": {}}

# Display Session Inspector in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Active Session Payload")
st.sidebar.json(st.session_state.chatbot_state)

# ==============================================================================
# 6. MAIN CHAT STREAM
# ==============================================================================
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your response here..."):
    if not api_key:
        st.error("Please enter your Groq API Key in the left sidebar to proceed.")
        st.stop()
        
    client = Groq(api_key=api_key)
    
    # Display user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Run Orchestrator Pipeline
    with st.spinner("Analyzing credit profile..."):
        bot_response = process_chat_message(client, prompt)
        
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.chat_message("assistant").write(bot_response)
    st.rerun()
