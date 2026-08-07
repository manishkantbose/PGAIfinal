import streamlit as st
import json
import re
import datetime
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

st.markdown("""
<style>
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        text-align: right;
        background-color: #e8f5e9;
        border-radius: 12px;
        padding: 8px 12px;
        margin-left: 20%;
    }
    
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

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in Streamlit Secrets!")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# ==============================================================================
# 2. MASTER CATALOGS & AUDIT LOGGING SYSTEM
# ==============================================================================
AUDIT_LOG_FILE = "audit_logs.jsonl"

def record_audit_log(event_type: str, payload: dict):
    """Appends an audit log entry with timestamp to a persistent JSONL file."""
    log_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event_type": event_type,
        "data": payload
    }
    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

EARLY_APPROVE_THRESH = 0.30
EARLY_DECLINE_THRESH = 0.70
FINAL_DECISION_THRESH = 0.44

MAX_HARD_STOP_LTV = 100.0    # Hard rejection if LTV > 100%
TARGET_POLICY_LTV = 80.0     # Safe underwriting baseline target
MIN_MONTHLY_INCOME = 1000.0  # Minimum income floor
MIN_AGE = 18
MAX_AGE = 60

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
    "requested_loan_amount": "Requested Loan Amount",
    "vehicle_price": "On-Road Vehicle Price",
    "Employment_Type": "Occupation / Employment Type (e.g. Salaried, Business, Student)",
    "monthly_income": "Monthly Income",
    "age": "Age",
    "pincode": "Pincode",
    "has_active_loans": "Active Loans / Existing EMIs status",
    "Make_Code": "Vehicle Brand/Make",
    "Model_Variant": "Vehicle Model/Variant"
}

# ==============================================================================
# 3. UNDERWRITING MODEL & POLICY ENGINE LOGIC
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

def check_over_invoicing(p1_data: dict) -> bool:
    price = float(p1_data.get('vehicle_price', 0))
    make = p1_data.get('Make_Code')
    variant = p1_data.get('Model_Variant')
    
    matched = PRICE_BENCHMARK_MASTER[
        (PRICE_BENCHMARK_MASTER['Make_Code'] == make) & 
        (PRICE_BENCHMARK_MASTER['Model_Variant'] == variant)
    ]
    if matched.empty and make:
        matched = PRICE_BENCHMARK_MASTER[PRICE_BENCHMARK_MASTER['Make_Code'] == make]
        
    if not matched.empty:
        max_allowed = float(matched['Max_Price'].max())
        return price > max_allowed
    return False

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
    loan_to_salary = req_loan / (salary + 1.0)
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

def predict_risk(model, df_features: pd.DataFrame, model_columns: list, is_over_invoiced: bool = False) -> float:
    aligned_df = df_features.reindex(columns=model_columns, fill_value=0).astype(np.float32)
    base_risk = float(model.predict_proba(aligned_df)[0][1])
    penalty = 0.15 if is_over_invoiced else 0.0
    return min(1.0, base_risk + penalty)

def evaluate_hard_policy_stops(p1_data: dict, p2_data: dict = None) -> tuple:
    reasons = []
    req_loan = float(p1_data.get('requested_loan_amount', 0))
    price = float(p1_data.get('vehicle_price', 1))
    ltv = (req_loan / price * 100.0) if price > 0 else 0.0

    if ltv > MAX_HARD_STOP_LTV:
        reasons.append(
            f"Loan-to-Value (LTV) ratio is **{ltv:.1f}%**, which exceeds our maximum permissible policy limit of **{MAX_HARD_STOP_LTV:.0f}%** "
            f"(Loan requested: ₹{req_loan:,.0f} vs Vehicle Price: ₹{price:,.0f})."
        )

    if p2_data:
        income = float(p2_data.get('monthly_income', 0))
        age = p2_data.get('age')
        emp_type = p2_data.get('Employment_Type', '')

        if income < MIN_MONTHLY_INCOME:
            reasons.append(
                f"Reported monthly income of **₹{income:,.0f}** is below our minimum policy floor of **₹{MIN_MONTHLY_INCOME:,.0f}**."
            )

        if age is not None and (int(age) < MIN_AGE or int(age) > MAX_AGE):
            reasons.append(
                f"Applicant age (**{age} years**) falls outside our eligible policy age bracket ({MIN_AGE} to {MAX_AGE} years)."
            )

        if emp_type in ['STU', 'NONEARNMEM']:
            reasons.append(
                "Employment category falls under non-earning status (Student / Non-Working Member), requiring a primary earning co-applicant."
            )

    return len(reasons) > 0, reasons

# ==============================================================================
# 4. COUNTER-OFFER & REASON GENERATOR LOGIC
# ==============================================================================
def calculate_max_eligible_loan(model, p1_data: dict, p2_data: dict) -> dict:
    req_loan = float(p1_data.get('requested_loan_amount', 0))
    vehicle_price = float(p1_data.get('vehicle_price', 1))
    is_over_invoiced = check_over_invoicing(p1_data)
    
    max_policy_loan = vehicle_price * (TARGET_POLICY_LTV / 100.0)
    search_ceiling = min(req_loan, max_policy_loan)
    
    low, high, max_approved = 0.0, search_ceiling, 0.0
    
    for _ in range(12):
        mid = (low + high) / 2.0
        test_p1 = p1_data.copy()
        test_p1['requested_loan_amount'] = mid
        df_feat = build_model_features(test_p1, p2_data)
        risk = predict_risk(model, df_feat, MODEL_COLUMNS, is_over_invoiced=is_over_invoiced)
        
        if risk <= FINAL_DECISION_THRESH:
            max_approved = mid
            low = mid
        else:
            high = mid
            
    is_partial_possible = (max_approved >= (0.30 * vehicle_price)) and (max_approved >= 10000)
    
    return {
        "requested_loan": req_loan,
        "max_eligible_loan": max_approved,
        "is_partial_possible": is_partial_possible,
        "max_policy_cap": max_policy_loan
    }

def generate_explained_decline_analysis(p1_data: dict, p2_data: dict, hard_stop_reasons: list = None) -> tuple:
    reasons = hard_stop_reasons.copy() if hard_stop_reasons else []
    improvements = []

    req_loan = float(p1_data.get('requested_loan_amount', 0))
    price = float(p1_data.get('vehicle_price', 1))
    income = float(p2_data.get('monthly_income', 0)) if p2_data else 0.0
    active_loans = int(p2_data.get('has_active_loans', 0)) if p2_data else 0

    ltv = (req_loan / price) * 100.0 if price > 0 else 0.0
    max_target_loan = price * (TARGET_POLICY_LTV / 100.0)
    extra_downpayment_needed = max(0.0, req_loan - max_target_loan)

    if ltv > TARGET_POLICY_LTV:
        if ltv <= MAX_HARD_STOP_LTV:
            reasons.append(f"LTV ratio of **{ltv:.1f}%** exceeds target policy limit of **{TARGET_POLICY_LTV:.0f}%**.")
        if extra_downpayment_needed > 0:
            improvements.append(
                f"**Increase Down Payment:** Pay an extra **₹{extra_downpayment_needed:,.0f}** down payment "
                f"to adjust the loan to ₹{max_target_loan:,.0f} (80% LTV)."
            )

    if income > 0 and (req_loan / income) > 3.0:
        reasons.append(f"Requested loan amount is high relative to net monthly income ({req_loan / income:.1f}x monthly income).")
        improvements.append("**Add Co-Applicant:** Include an earning co-applicant to increase total recognized household income.")

    if check_over_invoicing(p1_data):
        reasons.append("Vehicle quotation exceeds maximum market benchmark price for this variant.")
        improvements.append("**Review Quotation:** Verify dealer invoice price against standard market benchmark.")

    if active_loans == 1:
        reasons.append("Active prior loan obligations increase overall credit risk.")
        improvements.append("**Clear Existing Debt:** Pay off active EMIs or short-term loans to reduce debt burden.")

    if not reasons:
        reasons.append("Overall risk score exceeded underwriting thresholds.")

    if not improvements:
        improvements.append(f"**Lower Loan Request:** Lower loan request to ₹{req_loan * 0.85:,.0f} to improve eligibility.")

    return reasons, improvements

# ==============================================================================
# 5. CONTEXT-AWARE EXTRACTION & DIALOGUE HELPERS
# ==============================================================================
def extract_all_slots(user_input: str, current_state: dict, last_assistant_message: str = "") -> dict:
    system_prompt = f"""
    You are an AI loan entity extraction engine.
    Extract slots accurately from the user's message, taking into account what the assistant just asked them.
    
    CONVERSATIONAL CONTEXT RULE:
    - Assistant's last prompt to user: "{last_assistant_message}"
    - Current collected state: {json.dumps(current_state)}
    - IF the assistant explicitly asked for a specific field (e.g., "On-Road Vehicle Price") and the user responds with a standalone number or currency string (e.g., "250000 rs", "2.5 lakh", "250000"), YOU MUST MAP THIS VALUE directly to that requested field ("vehicle_price").

    NUMERIC & CURRENCY EXTRACTION RULES:
    - Parse numeric values accurately from Indian formats:
      - "250000 rs" / "250000" -> 250000.0
      - "2.5 lakh" / "2.5L" / "2.5 lakhs" -> 250000.0
      - "250k" -> 250000.0
    
    MANDATORY ENUM MAPPINGS:
    - Make_Code choice must be mapped to one of: {MAKE_MASTER}
    - Model_Variant choice must be mapped to closest variant from: {VARIANT_MASTER}
    - Product_Code choice must be one of: {PRODUCT_CODE_MASTER}
    - Employment_Type choice must be one of: {EMPLOYMENT_TYPE_MASTER}
      ('salaried' -> 'SAL', 'self employed'/'business' -> 'SEP', 'student' -> 'STU', 'farmer'/'agri' -> 'AGR', 'pensioner' -> 'PEN')
    - has_active_loans: 1 if user indicates active loans/EMIs, 0 if user explicitly states no active loans/EMIs, null if unmentioned.

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

def format_prompt_question(missing_mandatory: list, missing_optional: list) -> str:
    mandatory_str = ", ".join([f"**{FIELD_LABELS[f]}**" for f in missing_mandatory])
    msg = f"To proceed with your application, please provide your {mandatory_str}."
    if missing_optional:
        optional_str = ", ".join([f"**{FIELD_LABELS[f]}**" for f in missing_optional])
        msg += f"\n\n*(Optionally, you can also mention your {optional_str} to help us give you a better offer)*"
    return msg

# ==============================================================================
# 6. PIPELINE ORCHESTRATION & STATE MACHINE
# ==============================================================================
def process_chat_message(user_input: str):
    state = st.session_state.chatbot_state
    
    last_assistant_msg = ""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            last_assistant_msg = msg["content"]
            break

    extracted = extract_all_slots(
        user_input=user_input, 
        current_state={"p1": state["p1_data"], "p2": state["p2_data"]},
        last_assistant_message=last_assistant_msg
    )
    
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
    
    # --------------------------------------------------------------------------
    # PHASE 1 EVALUATION
    # --------------------------------------------------------------------------
    if state["step"] == "PHASE_1_COLLECTION":
        p1_mandatory = ['requested_loan_amount', 'vehicle_price']
        p1_optional = ['Make_Code', 'Model_Variant']
        
        missing_p1_mand = [f for f in p1_mandatory if p1_data.get(f) is None]
        missing_p1_opt = [f for f in p1_optional if p1_data.get(f) is None]
        
        if missing_p1_mand:
            return format_prompt_question(missing_p1_mand, missing_p1_opt)
        
        is_hard_stop, hard_reasons = evaluate_hard_policy_stops(p1_data)
        if is_hard_stop:
            state["step"] = "COMPLETED"
            offer = calculate_max_eligible_loan(ml_model, p1_data, p2_data)
            reasons, improvements = generate_explained_decline_analysis(p1_data, p2_data, hard_reasons)
            
            record_audit_log("LOAN_HARD_DECLINE", {
                "p1_data": p1_data,
                "p2_data": p2_data,
                "reasons": reasons,
                "counter_offer": offer
            })

            reasons_formatted = "\n".join([f"- {r}" for r in reasons])
            improvements_formatted = "\n".join([f"- {i}" for i in improvements])

            if offer["is_partial_possible"]:
                return (
                    f"Thank you for applying. We cannot approve **₹{p1_data['requested_loan_amount']:,.0f}** due to policy bounds.\n\n"
                    f"**Primary Decision Reason(s):**\n{reasons_formatted}\n\n"
                    f"🎉 However, you are **PRE-APPROVED** for an eligible loan of up to **₹{offer['max_eligible_loan']:,.0f}** (capped at 80% LTV).\n\n"
                    f"💡 **Recommended Action:**\n{improvements_formatted}"
                )
            else:
                return (
                    f"Thank you for applying. We are unable to approve your loan request of **₹{p1_data['requested_loan_amount']:,.0f}**.\n\n"
                    f"**Primary Decision Reason(s):**\n{reasons_formatted}\n\n"
                    f"💡 **How to Qualify:**\n{improvements_formatted}"
                )

        is_over_invoiced = check_over_invoicing(p1_data)
        p1_features = build_model_features(p1_data, p2_data)
        p1_risk = predict_risk(ml_model, p1_features, MODEL_COLUMNS, is_over_invoiced=is_over_invoiced)
        
        if p1_risk <= EARLY_APPROVE_THRESH:
            state["step"] = "COMPLETED"
            record_audit_log("PHASE_1_INSTANT_APPROVAL", {
                "p1_data": p1_data,
                "risk_score": p1_risk
            })
            return f"🎉 Excellent news! Your loan request of **₹{p1_data['requested_loan_amount']:,.0f}** for the **{p1_data.get('Model_Variant', 'vehicle')}** is **INSTANTLY PRE-APPROVED**!"
        elif p1_risk >= EARLY_DECLINE_THRESH:
            offer = calculate_max_eligible_loan(ml_model, p1_data, p2_data)
            state["step"] = "COMPLETED"
            reasons, improvements = generate_explained_decline_analysis(p1_data, p2_data)
            
            record_audit_log("PHASE_1_DECLINE", {
                "p1_data": p1_data,
                "risk_score": p1_risk,
                "reasons": reasons,
                "counter_offer": offer
            })

            reasons_formatted = "\n".join([f"- {r}" for r in reasons])
            improvements_formatted = "\n".join([f"- {i}" for i in improvements])

            if offer["is_partial_possible"]:
                return (
                    f"Thank you for applying. We cannot approve ₹{p1_data['requested_loan_amount']:,.0f}, but you are pre-approved for up to **₹{offer['max_eligible_loan']:,.0f}**.\n\n"
                    f"**Key Reason(s):**\n{reasons_formatted}\n\n"
                    f"💡 **Action Steps:**\n{improvements_formatted}"
                )
            else:
                return (
                    f"Thank you for applying. We are unable to approve your loan request of **₹{p1_data['requested_loan_amount']:,.0f}** at this time.\n\n"
                    f"**Primary Reason(s):**\n{reasons_formatted}\n\n"
                    f"💡 **How to Improve:**\n{improvements_formatted}"
                )
        else:
            state["step"] = "PHASE_2_COLLECTION"

    # --------------------------------------------------------------------------
    # PHASE 2 EVALUATION
    # --------------------------------------------------------------------------
    if state["step"] == "PHASE_2_COLLECTION":
        p2_mandatory = ['Employment_Type', 'monthly_income', 'age', 'pincode', 'has_active_loans']
        missing_p2_mand = [f for f in p2_mandatory if p2_data.get(f) is None]
        
        if missing_p2_mand:
            return format_prompt_question(missing_p2_mand, missing_optional=[])
        
        is_hard_stop, hard_reasons = evaluate_hard_policy_stops(p1_data, p2_data)
        is_over_invoiced = check_over_invoicing(p1_data)
        full_features = build_model_features(p1_data, p2_data)
        final_risk = predict_risk(ml_model, full_features, MODEL_COLUMNS, is_over_invoiced=is_over_invoiced)
        state["step"] = "COMPLETED"
        
        req = p1_data["requested_loan_amount"]
        is_approved = (final_risk < FINAL_DECISION_THRESH and not is_hard_stop)

        record_audit_log("FINAL_DECISION", {
            "status": "APPROVED" if is_approved else "DECLINED",
            "risk_score": final_risk,
            "p1_data": p1_data,
            "p2_data": p2_data,
            "hard_stop": is_hard_stop
        })

        if is_approved:
            return f"🎉 Congratulations! Your loan request of **₹{req:,.0f}** has been **FULLY APPROVED**."
        else:
            offer = calculate_max_eligible_loan(ml_model, p1_data, p2_data)
            max_l = offer["max_eligible_loan"]
            reasons, improvements = generate_explained_decline_analysis(p1_data, p2_data, hard_reasons if is_hard_stop else None)
            
            reasons_formatted = "\n".join([f"- {r}" for r in reasons])
            improvements_formatted = "\n".join([f"- {i}" for i in improvements])
            
            if offer["is_partial_possible"]:
                return (
                    f"Thank you for applying. While we cannot approve **₹{req:,.0f}**, "
                    f"based on your profile you are pre-approved for an eligible loan amount of up to **₹{max_l:,.0f}**.\n\n"
                    f"**Primary Key Reason(s):**\n{reasons_formatted}\n\n"
                    f"💡 **How to Qualify for Full Amount:**\n{improvements_formatted}"
                )
            else:
                return (
                    f"Thank you for applying. We are unable to approve your loan request of **₹{req:,.0f}** at this time.\n\n"
                    f"**Primary Key Reason(s):**\n{reasons_formatted}\n\n"
                    f"💡 **How to Improve Eligibility:**\n{improvements_formatted}"
                )

    return "Session complete."

def reset_application():
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Welcome to ABC Credit! Which vehicle model/variant are you looking to buy, what is its on-road price, and how much loan do you need?"}
    ]
    st.session_state.chatbot_state = {"step": "PHASE_1_COLLECTION", "p1_data": {}, "p2_data": {}}

# ==============================================================================
# 7. CHAT UI INTERFACE
# ==============================================================================
st.title("🚗 ABC Credit - Intelligent Loan Assistant")

if "chatbot_state" not in st.session_state:
    st.session_state.chatbot_state = {"step": "PHASE_1_COLLECTION", "p1_data": {}, "p2_data": {}}

if "messages" not in st.session_state:
    reset_application()

st.sidebar.header("⚙️ Application Controls")
if st.sidebar.button("🔄 Reset / Start Over"):
    reset_application()
    st.rerun()

with st.sidebar.expander("🛠️ Developer Debugger", expanded=False):
    st.json(st.session_state.chatbot_state)

with st.sidebar.expander("📜 Audit Logs Viewer", expanded=False):
    if st.button("Refresh Logs"):
        st.rerun()
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f.readlines()]
            if logs:
                st.dataframe(pd.DataFrame(logs))
            else:
                st.info("Log file is empty.")
    except FileNotFoundError:
        st.info("No audit logs recorded yet.")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

is_completed = st.session_state.chatbot_state.get("step") == "COMPLETED"
with st.sidebar.expander("📜 Audit Logs Viewer", expanded=False):
    admin_pass = st.text_input("Enter Admin Password", type="password")
    
    # Replace "your_secure_password" with your desired password or use st.secrets
    if admin_pass == "your_secure_password":
        if st.button("Refresh Logs"):
            st.rerun()
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                logs = [json.loads(line) for line in f.readlines()]
                if logs:
                    st.dataframe(pd.DataFrame(logs))
                else:
                    st.info("Log file is empty.")
        except FileNotFoundError:
            st.info("No audit logs recorded yet.")
    elif admin_pass:
        st.error("Incorrect password.")
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
