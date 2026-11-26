# ◧ AT-RISK INTELLIGENCE — Predict • Detect • Act

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=adityaranjandp28-web/at-risk-intelligence&branch=main&mainModule=streamlit_app.py)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B.svg)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, multi-dimensional **Academic and Workforce Risk Analytics System** engineered in Python using **Streamlit** and **Plotly**. 

Designed to replace static, retrospective reports with proactive diagnostic intelligence: identifying at-risk individuals early through attendance patterns, assessment scores, engagement indicators, and temporal decay trajectories.

---

## 🚀 Live Cloud Deployment (Zero Setup / Single Link)

You can launch and use the application directly in your web browser with **zero installation**:

👉 **[Launch Live Dashboard on Streamlit Cloud](https://share.streamlit.io/deploy?repository=adityaranjandp28-web/at-risk-intelligence&branch=main&mainModule=streamlit_app.py)**

*(Clicking the link or the badge above deploys the repository for free on Streamlit Community Cloud with a permanent web link).*

---

## ✨ Core Features & Capabilities

* **Dual Operational Modes:**
  * 🎓 **Education Mode:** Analyzes student attendance, CGPA, quiz scores, assignment backlogs, and failure histories.
  * 🏢 **Corporate Mode:** Evaluates employee performance, overtime workloads, engagement scores, and retention risks.
* **Empirical Composite Risk Scoring:**
  * Multivariate weighting model classifying cohorts into 4 distinct tiers: `Low`, `Moderate`, `High`, and `Critical`.
  * Customizable factor weights and sensitivity thresholds in real-time.
* **Interactive Diagnostic Views:**
  * **Executive Overview:** High-level KPI cards, attendance compliance distribution, and bivariate risk matrices.
  * **Longitudinal Trends:** 6-month / 12-week performance and attendance trajectory decay curves (spotting the midterm cliff).
  * **Multivariate Correlation Matrix:** Interactive heatmaps analyzing inter-variable dependencies.
  * **Actionable Case Triage Roster:** Prioritized intervention queue with counselor tracking statuses (*New*, *Contacted*, *In Progress*, *Resolved*).
* **Flexible Data Management:**
  * Built-in high-fidelity synthetic demo data (120 students & 120 employees).
  * Upload custom datasets via **CSV** or **Excel (.xlsx)** with automated column aliasing and smart schema mapping.
  * Export filtered cohorts and custom reports.

---

## 🛠️ Local Installation & Development

If you prefer to run the app locally on your machine:

### 1. Clone the repository
```bash
git clone https://github.com/adityaranjandp28-web/at-risk-intelligence.git
cd at-risk-intelligence
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run streamlit_app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

The repository includes a comprehensive 58-test suite validating the risk model heuristics, schema auto-mapping, and edge cases:

```bash
python test_at_risk.py
```

---

## 📂 Project Architecture

```
at-risk-intelligence/
├── at_risk_intelligence.py     # Core application logic, scoring models, and UI components
├── streamlit_app.py            # Streamlit Cloud deployment entrypoint
├── test_at_risk.py             # Comprehensive 58-test validation suite
├── requirements.txt            # Production dependencies
├── .streamlit/
│   └── config.toml             # Dark theme styling and server configuration
├── .gitignore                  # Git hygiene and exclusion rules
└── README.md                   # Documentation & deployment guide
```

---

## 👤 Author

* **Aditya Ranjan**
* GitHub: [@adityaranjandp28-web](https://github.com/adityaranjandp28-web)
* Institution: Computer Science & Engineering, Bennett University

---

## 📄 License
This project is licensed under the MIT License.
