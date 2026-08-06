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

# Custom CSS for chat layout & CTA styling
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
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
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
# 2. INDEPENDENT MASTER CATALOGS
# ==============================================================================
EARLY_APPROVE_THRESH = 0.15
EARLY_DECLINE_THRESH = 0.80
FINAL_DECISION_THRESH = 0.44

PRICE_BENCHMARK_MASTER = pd.DataFrame([
    {"Make_Code": "RAIDER", "Model_Variant": "RAIDER", "Min_Price": 114698.26, "Median_Price": 118997.16, "Max_Price": 130714.53, "Approved_Samples": 17654},
    {"Make_Code": "MOPEDS", "Model_Variant": "XL 100CC", "Min_Price": 52515.94, "Median_Price": 75027.28, "Max_Price": 84301.12, "Approved_Samples": 15718},
    {"Make_Code": "APACHE", "Model_Variant": "160 DISC", "Min_Price": 150542.48, "Median_Price": 154555.12, "Max_Price": 168522.56, "Approved_Samples": 15541},
    {"Make_Code": "NTORQ", "Model_Variant": "125 CC", "Min_Price": 114845.93, "Median_Price": 123306.93, "Max_Price": 136088.58, "Approved_Samples": 11301},
    {"Make_Code": "JUPITER", "Model_Variant": "110 CC", "Min_Price": 92559.39, "Median_Price": 105592.22, "Max_Price": 118442.86, "Approved_Samples": 10653},
    {"Make_Code": "SPORT", "Model_Variant": "SPORT", "Min_Price": 73652.69, "Median_Price": 75839.41, "Max_Price": 89287.53, "Approved_Samples": 5641},
    {"Make_Code": "RADEON", "Model_Variant": "RADEON", "Min_Price": 83145.97, "Median_Price": 94395.00, "Max_Price": 105997.72, "Approved_Samples": 4845},
    {"Make_Code": "JUPITER", "Model_Variant": "125 CC DISC", "Min_Price": 109002.99, "Median_Price": 118341.78, "Max_Price": 127220.82, "Approved_Samples": 4625},
    {"Make_Code": "ZEST", "Model_Variant": "SCOOTY", "Min_Price": 88000.54, "Median_Price": 95596.18, "Max_Price": 100705.37, "Approved_Samples": 3509},
    {"Make_Code": "TVS", "Model_Variant": "EBIKE", "Min_Price": 144362.07, "Median_Price": 154534.52, "Max_Price": 170006.91, "Approved_Samples": 3365},
    {"Make_Code": "APACHE", "Model_Variant": "160 DRUM", "Min_Price": 145911.10, "Median_Price": 147830.30, "Max_Price": 152054.85, "Approved_Samples": 3246},
    {"Make_Code": "JUPITER", "Model_Variant": "125CC DISC", "Min_Price": 116827.95, "Median_Price": 125844.79, "Max_Price": 131851.67, "Approved_Samples": 3212},
    {"Make_Code": "JUPITER", "Model_Variant": "125 CC DRUM", "Min_Price": 103997.90, "Median_Price": 112992.30, "Max_Price": 120617.33, "Approved_Samples": 1559},
    {"Make_Code": "CITY PLUS", "Model_Variant": "STAR CITY PLUS", "Min_Price": 92904.74, "Median_Price": 96257.48, "Max_Price": 104998.85, "Approved_Samples": 627},
    {"Make_Code": "RONIN", "Model_Variant": "RONIN", "Min_Price": 179217.49, "Median_Price": 202401.60, "Max_Price": 220911.68, "Approved_Samples": 577},
    {"Make_Code": "APACHE", "Model_Variant": "APACHE 180 CC", "Min_Price": 160376.92, "Median_Price": 162239.56, "Max_Price": 170382.76, "Approved_Samples": 337},
    {"Make_Code": "APACHE", "Model_Variant": "RTR 200CC", "Min_Price": 168748.18, "Median_Price": 181759.29, "Max_Price": 190904.13, "Approved_Samples": 187},
    {"Make_Code": "APACHE", "Model_Variant": "APACHE RTR 310", "Min_Price": 287202.31, "Median_Price": 311024.17, "Max_Price": 334139.41, "Approved_Samples": 29},
    {"Make_Code": "APACHE", "Model_Variant": "APACHE RR 310", "Min_Price": 315333.53, "Median_Price": 322294.39, "Max_Price": 345082.11, "Approved_Samples": 25},
    {"Make_Code": "SCOOTY", "Model_Variant": "SCOOTY PEP PLUS", "Min_Price": 79125.23, "Median_Price": 85898.92, "Max_Price": 97028.13, "Approved_Samples": 8},
    {"Make_Code": "APACHE", "Model_Variant": "165 CC", "Min_Price": 179993.95, "Median_Price": 179993.95, "Max_Price": 179993.95, "Approved_Samples": 1}
])

MAKE_MASTER = PRICE_BENCHMARK_MASTER["Make_Code"].unique().tolist()
VARIANT_MASTER = PRICE_BENCHMARK_MASTER["Model_Variant"].unique().tolist()
EMPLOYMENT_TYPE_MASTER = ["SAL", "SEP", "STU", "AGR", "PEN", "NONEARNMEM", "NREGI", "NPP"]
PRODUCT_CODE_MASTER = ["MC", "SC", "MO", "EB"]

PREMIUM_WORDS = {
    "4V": 2, "310": 3, "350": 3, "200": 1, "165 RP": 2,
    "2CH": 2, "1CH": 1, "ABS": 2, "DISC": 1,
    "R MODE": 2, "RACE": 2, "RACE XP": 2, "SUPER SQUAD": 2,
    "SPL EDITION": 2, "WINNER EDITION": 1, "MATTE SERIES": 1,
    "CLASSIC": 1, "XT": 1, "ST": 2, "ALLOY": 1, "BSVI": 1,
    "OBDIIA": 1, "EV": 3, "IQUBE": 3
}

FIELD_LABELS = {
    "requested_loan_amount": "requested loan amount",
    "vehicle_price": "on-road vehicle price",
    "Employment_Type": "occupation type (e.g., Salaried, Self-Employed, Student, Agriculture)",
    "monthly_income": "monthly income",
    "age": "age",
    "pincode": "area pincode",
    "has_active_loans": "active loans status (whether you currently have existing active loans)"
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
# 4. UNIFIED EXTRACTION & UNDERWRITING LOGIC
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

def extract_all_slots(user_input: str) -> dict:
    """Unified slot extractor capturing both Phase 1 and Phase 2 entities simultaneously."""
    system_prompt = f"""
    You are an AI automotive loan entity extraction engine. Extract all relevant details from user text.
    
    MANDATORY ENUM MAPPINGS:
    - Make_Code choice must be mapped to one of: {MAKE_MASTER}
    - Model_Variant choice must be mapped to closest variant from: {VARIANT_MASTER}
    - Product_Code choice must be one of: {PRODUCT_CODE_MASTER}
    - Employment_Type choice must be one of: {EMPLOYMENT_TYPE_MASTER}
      (Map 'salaried' -> 'SAL', 'self employed'/'business' -> 'SEP', 'student' -> 'STU', 'farmer'/'agri' -> 'AGR', 'pensioner' -> 'PEN')
    - has_active_loans: Set to 1 if user indicates active loans/EMIs, 0 if user explicitly states no active loans/EMIs, null if unmentioned.

    Return ONLY JSON:
    {{
      "requested_loan_amount": FLOAT_OR_NULL,
      "vehicle_price": FLOAT_OR_NULL,
      "Product_Code": "ENUM_OR_NULL",
      "Make_Code": "ENUM_OR_NULL",
      "Model_Variant": "ENUM_OR_NULL",
      "model_description": "STRING_OR_NULL",
      "Employment_Type": "ENUM_OR_NULL",
      "monthly_income": FLOAT_OR_NULL,
      "age": INT_OR_NULL,
      "pincode": "STRING_OR_NULL",
      "has_active_loans": INT_0_1_OR_NULL
    }}
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {}

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

def build_pinpoint_question(missing_fields: list) -> str:
    """Generates precise, dynamic follow-up questions for missing fields only."""
    readable_missing = [FIELD_LABELS.get(f, f) for f in missing_fields]
    
    if len(readable_missing) == 1:
        items_str = f"your **{readable_missing[0]}**"
    elif len(readable_missing) == 2:
        items_str = f"your **{readable_missing[0]}** and **{readable_missing[1]}**"
    else:
        items_str = ", ".join([f"**{item}**" for item in readable_missing[:-1]]) + f", and **{readable_missing[-1]}**"
        
    return f"To proceed with your application, please provide {items_str}."

# ==============================================================================
# 5. STATE MACHINE & PIPELINE ORCHESTRATOR
# ==============================================================================
def process_chat_message(user_input: str):
    state = st.session_state.chatbot_state
    
    # 1. Unified extraction pass across all fields
    extracted = extract_all_slots(user_input)
    
    # Route extracted values to appropriate phase store
    p1_keys = ['requested_loan_amount', 'vehicle_price', 'Product_Code', 'Make_Code', 'Model_Variant', 'model_description']
    p2_keys = ['Employment_Type', 'monthly_income', 'age', 'pincode', 'has_active_loans']
    
    for k, v in extracted.items():
        if v is not None:
            if k in p1_keys:
                state["p1_data"][k] = v
            elif k in p2_keys:
                state["p2_data"][k] = v

    p1_data = state["p1_data"]
    p2_data = state["p2_data"]
    
    # 2. Phase 1 Evaluation Logic
    if state["step"] == "PHASE_1_COLLECTION":
        p1_mandatory = ['requested_loan_amount', 'vehicle_price']
        missing_p1 = [field for field in p1_mandatory if p1_data.get(field) is None]
        
        if missing_p1:
            return build_pinpoint_question(missing_p1)
        
        # Mandatory Phase 1 fields collected -> Assess Phase 1 Risk
        p1_features = build_model_features(p1_data, p2_data)
        p1_risk = predict_risk(ml_model, p1_features, MODEL_COLUMNS)
        
        if p1_risk <= EARLY_APPROVE_THRESH:
            state["step"] = "COMPLETED"
            return f"🎉 Excellent news! Your loan request of **₹{p1_data['requested_loan_amount']:,.0f}** for the **{p1_data.get('Model_Variant', 'vehicle')}** is **INSTANTLY PRE-APPROVED**!"
        elif p1_risk >= EARLY_DECLINE_THRESH:
            offer = calculate_max_eligible_loan(ml_model, p1_data, p2_data)
            state["step"] = "COMPLETED"
            return f"Thank you for applying. We cannot approve ₹{p1_data['requested_loan_amount']:,.0f}, but you are pre-approved for up to **₹{offer['max_eligible_loan']:,.0f}**."
        else:
            state["step"] = "PHASE_2_COLLECTION"

    # 3. Phase 2 Evaluation Logic
    if state["step"] == "PHASE_2_COLLECTION":
        p2_mandatory = ['Employment_Type', 'monthly_income', 'age', 'pincode', 'has_active_loans']
        missing_p2 = [field for field in p2_mandatory if p2_data.get(field) is None]
        
        if missing_p2:
            return build_pinpoint_question(missing_p2)
        
        # All Phase 2 fields present -> Run full underwriting
        full_features = build_model_features(p1_data, p2_data)
        final_risk = predict_risk(ml_model, full_features, MODEL_COLUMNS)
        state["step"] = "COMPLETED"
        
        req = p1_data["requested_loan_amount"]
        if final_risk < FINAL_DECISION_THRESH:
            return f"🎉 Congratulations! Your loan request of **₹{req:,.0f}** has been **FULLY APPROVED**."
        else:
            offer = calculate_max_eligible_loan(ml_model, p1_data, p2_data)
            max_l = offer["max_eligible_loan"]
            if offer["is_partial_possible"]:
                return f"Thank you for applying. While we cannot approve ₹{req:,.0f}, based on your profile you are pre-approved for up to **₹{max_l:,.0f}**. You can start a new application for this amount."
            else:
                return f"Thank you for applying. We are unable to approve your loan request of ₹{req:,.0f} at this time."

    return "Session complete."

def reset_application():
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome to ABC Credit! Which vehicle model/variant are you looking to buy, what is its on-road price, and how much loan do you need?"}
    ]
    st.session_state.chatbot_state = {"step": "PHASE_1_COLLECTION", "p1_data": {}, "p2_data": {}}

# ==============================================================================
# 6. UI LAYOUT & CHAT INTERFACE
# ==============================================================================
st.title("🚗 ABC Credit - Intelligent Loan Assistant")

# Initialize Session State
if "chatbot_state" not in st.session_state:
    st.session_state.chatbot_state = {"step": "PHASE_1_COLLECTION", "p1_data": {}, "p2_data": {}}

if "messages" not in st.session_state:
    reset_application()

# Clean Sidebar Controls
st.sidebar.header("⚙️ Application Controls")
if st.sidebar.button("🔄 Reset / Start Over"):
    reset_application()
    st.rerun()

# Developer Debugger (Hidden by default)
with st.sidebar.expander("🛠️ Developer Debugger", expanded=False):
    st.json(st.session_state.chatbot_state)

# Render Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# End-of-Flow CTA vs Chat Input
is_completed = st.session_state.chatbot_state.get("step") == "COMPLETED"

if is_completed:
    st.markdown("---")
    st.success("✅ Application process finished.")
    if st.button("🚀 Start New Loan Application", type="primary"):
        reset_application()
        st.rerun()
else:
    if prompt := st.chat_input("Type your response here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Processing response..."):
            bot_response = process_chat_message(prompt)
            
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.chat_message("assistant").write(bot_response)
        st.rerun()
