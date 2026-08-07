# ABC Credit - AI Conversational Loan Underwriting Assistant

An intelligent, two-phase conversational AI chatbot designed for automated vehicle loan underwriting and credit assessment. The system extracts application data from free-form user responses, validates loan parameters against policy guardrails, detects vehicle over-invoicing, computes ML-based risk scores, and delivers explainable decisions along with dynamic counter-offers.

---

## 🔗 Live Demos
- **Version 3 (Latest):** [https://chatbotforabccredit.streamlit.app/](https://chatbotforabccredit.streamlit.app/)
- **Version 2:** [https://abccreditchatbot.streamlit.app/](https://abccreditchatbot.streamlit.app/)

---

## ✨ Key Features

### 1. Conversational Slot Extraction (Groq LLM)
- Uses **Groq API (`llama-3.3-70b-versatile`)** to parse natural language user inputs into structured JSON parameters.
- Converts Indian numerical and currency formats (e.g., `2.5L`, `250k`, `250000 rs`) automatically.
- Maps vehicle makes, variants, product codes, and employment types to standard master catalogs.

### 2. Two-Phase Underwriting Pipeline
- **Phase 1 (Vehicle & Loan Request):** Collects vehicle details, on-road price, and loan amount requested. Instantly approves low-risk candidates or declines hard-policy violations without cluttering users with extra steps.
- **Phase 2 (Applicant Profile & Debt):** For intermediate-risk profiles, gathers income, age, pincode, employment type, and existing loan obligations for a comprehensive evaluation.

### 3. Policy Guardrails & Over-Invoicing Detection
- **Hard Policy Stops:** Automatically flags/declines applications violating core constraints:
  - Loan-to-Value (LTV) $> 100\%$
  - Monthly Income $< ₹1,000$
  - Age outside $18–60$ years
  - Non-earning employment profiles (e.g., Student, Non-Working Member)
- **Market Price Benchmark Check:** Cross-checks quoted vehicle price against a master database (`PRICE_BENCHMARK_MASTER`) to flag potential dealer over-invoicing and apply appropriate risk penalties.
- **Premium Vehicle Scoring:** Evaluates model keywords (e.g., `310`, `ABS`, `EV`, `RACE XP`) and engine displacement (CC) to adjust underwriting context.

### 4. ML Risk Scoring & Smart Counter-Offer Engine
- **Risk Assessment:** Uses a `HistGradientBoostingClassifier` trained on feature sets like LTV, Loan-to-Salary, Premium Score, and credit indicators to generate a calibrated risk score.
- **Explainable Decisions:** Rejection outcomes include detailed breakdowns explaining *why* an application was declined.
- **Binary Search Counter-Offers:** If the full requested loan cannot be approved, a binary search algorithm determines the highest eligible loan amount (at or below policy limits like $80\%$ LTV) and offers actionable advice on down payment adjustments.

### 5. Audit Logging & Observability
- All pipeline events (slot extraction, policy checks, ML probability outputs, and final decisions) are logged locally to `audit_logs.jsonl` with timestamps and session IDs for compliance and review.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Machine Learning & Analytics:** `scikit-learn` (`HistGradientBoostingClassifier`), `pandas`, `numpy`
- **UI Framework:** Streamlit
- **Logging:** JSON Lines (`.jsonl`) audit framework

---

## 🚀 Running Locally

### Prerequisites
Make sure you have a Groq API Key. Set it as an environment variable or enter it when prompted in the application:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
