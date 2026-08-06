import streamlit as st
import json
import re
import numpy as np
import pandas as pd
from groq import Groq
from sklearn.ensemble import HistGradientBoostingClassifier

# ==============================================================================
# 1. PAGE SETUP & CSS CHAT ALIGNMENT
# ==============================================================================
st.set_page_config(
    page_title="ABC Credit - Vehicle Loan Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS to align User messages to the RIGHT and Bot messages to the LEFT
st.markdown("""
<style>
    /* User chat bubble styling (Right Aligned) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        text-align: right;
        background-color: #e8f5e9;
        border-radius: 12px;
        padding: 8px 12px;
        margin-left: 20%;
    }
    
    /* Assistant chat bubble styling (Left Aligned) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        flex-direction: row;
        text-align: left;
        background-color: #f1f3f4;
        border-radius: 12px;
        padding: 8px 12px;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

# Fetch API Key from secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in Streamlit Secrets! Please add it to `.streamlit/secrets.toml`.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# ==============================================================================
# 2. MASTER DATA CATALOGS (Including Variant & Pricing Ranges)
# ==============================================================================
EARLY_APPROVE_THRESH = 0.15
EARLY_DECLINE_THRESH = 0.80
FINAL_DECISION_THRESH = 0.44

VARIANT_CATALOG = [
    {"Make_Code": "RAIDER", "Model_Variant": "RAIDER", "Min": 114698, "Median": 118997, "Max": 130714},
    {"Make_Code": "MOPEDS", "Model_Variant": "XL 100CC", "Min": 52515, "Median": 75027, "Max": 84301},
    {"Make_Code": "APACHE", "Model_Variant": "160 DISC", "Min": 150542, "Median": 154555, "Max": 168522},
    {"Make_Code": "NTORQ", "Model_Variant": "125 CC", "Min": 114845, "Median": 123306, "Max": 136088},
    {"Make_Code": "JUPITER", "Model_Variant": "110 CC", "Min": 92559, "Median": 105592, "Max": 118442},
    {"Make_Code": "SPORT", "Model_Variant": "SPORT", "Min": 73652, "Median": 75839, "Max": 89287},
    {"Make_Code": "RADEON", "Model_Variant": "RADEON", "Min": 83145, "Median": 94395, "Max": 105997},
    {"Make_Code": "JUPITER", "Model_Variant": "125 CC DISC", "Min": 109002, "Median": 118341, "Max": 127220},
    {"Make_Code": "ZEST", "Model_Variant": "SCOOTY", "Min": 88000, "Median": 95596, "Max": 100705},
    {"Make_Code": "TVS", "Model_Variant": "EBIKE", "Min": 144362, "Median": 154534, "Max": 170006},
    {"Make_Code": "APACHE", "Model_Variant": "160 DRUM", "Min": 145911, "Median": 147830, "Max": 152054},
    {"Make_Code": "JUPITER", "Model_Variant": "125CC DISC", "Min": 116827, "Median": 125844, "Max": 131851},
    {"Make_Code": "JUPITER", "Model_Variant": "125 CC DRUM", "Min": 103997, "Median": 112992, "Max": 120617},
    {"Make_Code": "CITY PLUS", "Model_Variant": "STAR CITY PLUS", "Min": 92904, "Median": 96257, "Max": 104998},
    {"Make_Code": "RONIN", "Model_Variant": "RONIN", "Min": 179217, "Median": 202401, "Max": 220911},
    {"Make_Code": "APACHE", "Model_Variant": "APACHE 180 CC", "Min": 160376, "Median": 162239, "Max": 170382},
    {"Make_Code": "APACHE", "Model_Variant": "RTR 200CC", "Min": 168748, "Median": 181759, "Max": 190904},
    {"Make_Code": "APACHE", "Model_Variant": "APACHE RTR 310", "Min": 287202, "Median": 311024, "Max": 334139},
    {"Make_Code": "APACHE", "Model_Variant": "APACHE RR 310", "Min": 315333, "Median": 322294, "Max": 345082},
    {"Make_Code": "SCOOTY", "Model_Variant": "SCOOTY PEP PLUS", "Min": 79125, "Median": 85898, "Max": 97028},
    {"Make_Code": "APACHE", "Model_Variant": "165 CC", "Min": 179993, "Median": 179993, "Max": 179993}
]

VALID_VARIANTS = list(set([item["Model_Variant"] for item in VARIANT_CATALOG]))

MASTER_DATA = {
    "Employment_Type": ["SAL", "SEP", "STU", "AGR", "PEN", "NONEARNMEM", "NREGI", "NPP"],
    "Product_Code": ["MC", "SC", "MO", "EB"],
    "Make_Code": list(set([item["Make_Code"] for item in VARIANT_CATALOG])),
    "Model_Variant": VALID_VARIANTS
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
# 3. MOCK UNDERWRITING MODEL
# ==============================================================================
@st.cache_resource
def load_underwriting_model():
    cols = [
        'Loan_Amount', 'Vehicle_Price', 'LTV', 'Loan_to_Salary', 
        'Log_Loan_Amount', 'Premium_Score', 'Age', 'Monthly_Income', 
        'Has_Active_Loans', 'Employment_Type_SAL', 'Employment_Type_SEP', 
        'Product_Code_MC', 'Product_Code_SC', 'Make_Code_APACHE', 'Make_Code_TVS'
    ]
    X_dummy = pd.DataFrame(np.random.rand(100, len(cols)), columns=cols)
    y_dummy = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_dummy, y_dummy)
    return model, cols

ml_model, MODEL_COLUMNS = load_underwriting_model()

# ==============================================================================
# 4. EXTRACTION & UNDERWRITING LOGIC
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

def extract_phase1_slots(user_input: str) -> dict:
    system_prompt = f"""
    You are an automotive entity extraction engine. Extract vehicle features and loan amount from user text.
    
    MANDATORY ENUM CONSTRAINTS:
    - Make_Code must be EXACTLY ONE of: {MASTER_DATA['Make_Code']}
    - Model_Variant must be mapped to the closest canonical variant from: {MASTER_DATA['Model_Variant']}
    - Product_Code choices: MC (Motorcycle), SC (Scooter), MO (Moped), EB (EV)
    
    Return ONLY JSON:
    {{
      "requested_loan_amount": FLOAT_OR_NULL,
      "vehicle_price": FLOAT_OR_NULL,
      "Product_Code": "ENUM_OR_NULL",
      "Make_Code": "ENUM_OR_NULL",
      "Model_Variant": "ENUM_OR_NULL",
      "model_description": "STRING_OR_NULL"
    }}
    """
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

def extract_phase2_slots(user_input: str) -> dict:
    system_prompt = """
    Extract customer demographics and map work/job description to Employment_Type.
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
    response = groq_client.chat.completions.create(
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
# 5. STATE MACHINE & PIPELINE ORCHESTRATOR
# ==============================================================================
def process_chat_message(user_input: str):
    state = st.session_state.chatbot_state
    step = state.get("step", "PHASE_1_COLLECTION")
    
    if step == "PHASE_1_COLLECTION":
        extracted = extract_phase1_slots(user_input)
        for k, v in extracted.items():
            if v is not None:
                state["p1_data"][k] = v
                
        p1_data = state["p1_data"]
        
        # Check if missing required slots
        missing = []
        if not p1_data.get('requested_loan_amount'): missing.append('requested_loan_amount')
        if not p1_data.get('vehicle_price'): missing.append('vehicle_price')
        
        if missing:
            if 'vehicle_price' in missing and p1_data.get('requested_loan_amount'):
                bot_msg = f"Got it, a loan request of ₹{p1_data['requested_loan_amount']:,.0f}! Which vehicle model/variant are you looking to buy, and what is its expected on-road price?"
            else:
                bot_msg = "Please specify both the requested loan amount and the on-road vehicle price to proceed."
            return bot_msg
        
        # Phase 1 Fast Screening Check
        p1_features = build_model_features(p1_data)
        p1_risk = predict_risk(ml_model, p1_features, MODEL_COLUMNS)
        
        if p1_risk <= EARLY_APPROVE_THRESH:
            state["step"] = "COMPLETED"
            return f"🎉 Excellent news! Your loan request of ₹{p1_data['requested_loan_amount']:,.0f} for the **{p1_data.get('Model_Variant', 'vehicle')}** is **INSTANTLY PRE-APPROVED**!"
        elif p1_risk >= EARLY_DECLINE_THRESH:
            offer = calculate_max_eligible_loan(ml_model, p1_data, {})
            state["step"] = "COMPLETED"
            return f"Thank you for applying. We cannot approve ₹{p1_data['requested_loan_amount']:,.0f}, but you are pre-approved for up to **₹{offer['max_eligible_loan']:,.0f}**."
        else:
            state["step"] = "PHASE_2_COLLECTION"
            return "Vehicle & loan details verified! To complete full underwriting, what is your current occupation, monthly salary, age, and address pincode?"

    elif step == "PHASE_2_COLLECTION":
        extracted = extract_phase2_slots(user_input)
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
                return f"Thank you for applying. While we cannot approve ₹{req:,.0f}, based on your profile you are pre-approved for up to **₹{max_l:,.0f}**. You can update your application to ₹{max_l:,.0f} to proceed."
            else:
                return f"Thank you for applying. We are unable to approve your loan request of ₹{req:,.0f} at this time."

    return "Session complete. Click 'Reset Application' in the sidebar to start over."

# ==============================================================================
# 6. UI LAYOUT & CHAT INTERFACE
# ==============================================================================
st.title("🚗 ABC Credit - Intelligent Loan Assistant")

# Sidebar Controls
st.sidebar.header("⚙️ Controls")
if st.sidebar.button("🔄 Reset Application"):
    st.session_state.messages = []
    st.session_state.chatbot_state = {"step": "PHASE_1_COLLECTION", "p1_data": {}, "p2_data": {}}
    st.rerun()

# Sidebar Master Inspector
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Active Application Data")
st.sidebar.json(st.session_state.chatbot_state)

# Initialize Session Message State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome to ABC Credit! Which vehicle model/variant are you looking to buy, what is its on-road price, and how much loan do you need?"}
    ]

if "chatbot_state" not in st.session_state:
    st.session_state.chatbot_state = {"step": "PHASE_1_COLLECTION", "p1_data": {}, "p2_data": {}}

# Render Chat Stream
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Chat Input
if prompt := st.chat_input("Type your message here..."):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Process through pipeline
    with st.spinner("Processing request..."):
        bot_response = process_chat_message(prompt)
        
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.chat_message("assistant").write(bot_response)
    st.rerun()
