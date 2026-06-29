# 🌌 Explainable Multimodal Sensor Fusion for Asteroid Threat Assessment

A physics-informed deep learning framework that classifies Near-Earth Asteroid (NEA) threat levels by fusing optical, radar, and spectral observations — with built-in explainability for engineers, scientists, and decision-makers.

---

## 📊 Results at a Glance

| Metric | Score |
|--------|-------|
| Accuracy (5-fold CV) | **94.78% ± 0.26%** |
| F1-Score | **0.9391 ± 0.0035** |
| Expected Calibration Error | **0.0245** |
| H-G Physics Consistency | **96%** |
| Kepler Physics Consistency | **94%** |
| vs Rule-based baseline | **+7.5 pp improvement** |

**Per-Class Performance:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Low Threat | 95.2% | 96.1% | 95.6% |
| Medium Threat | 92.8% | 91.9% | 92.4% |
| High Threat | 94.1% | 95.7% | 94.9% |

---

## 🔍 What This Project Does

Real asteroid observations arrive in stages — optical telescopes first, then radar days later, then spectral data. Existing models either wait for complete data or ignore the physics of celestial mechanics entirely.

This framework:
- **Fuses 3 sensor types** (optical, radar, spectral) via cross-modal attention
- **Enforces physics constraints** — H-G photometry and Kepler's laws baked into training
- **Updates predictions sequentially** as sensors come in (T1 → T2 → T3)
- **Explains every prediction** at 3 levels: sensor attribution, physics scores, and plain-language summaries

---

## 🏗️ Architecture

```
INPUT LAYER
├── Optical Sensor (7 features): H magnitude, distance, orbital params
├── Radar Sensor (4 features): velocity, MOID, ranging data
└── Spectral Sensor (3 features): albedo, diameter, spectral type

ENCODER PATHWAYS (parallel)
├── Optical Encoder:  7 → 128 → 64
├── Radar Encoder:    4 → 128 → 64
└── Spectral Encoder: 3 → 128 → 64

CROSS-MODAL ATTENTION  →  64 fused features + sensor weights

PHYSICS VALIDATION LAYER
├── H-G Photometry consistency check
└── Kepler's Third Law verification

CLASSIFIER:  64 → 32 → 3  (Low / Medium / High threat)

OUTPUT
├── Threat classification
├── Confidence score (0–100%)
├── Sensor attribution (Optical X%, Radar Y%, Spectral Z%)
└── Physics consistency metrics
```

**Sequential Update Flow:**
- **T1** — Optical only → initial prediction, high uncertainty
- **T2** — Optical + Radar → refined prediction, lower uncertainty
- **T3** — All 3 sensors → final prediction, minimal uncertainty

---

## 📁 Project Structure

```
asteroid-threat-assessment/
├── src/
│   └── asteroid_threat_model.py   # Full pipeline: data + model + training + explainability
├── data/
│   ├── raw/                       # NASA CAD API data (fetched at runtime)
│   ├── processed/                 # real_asteroids.csv, synthetic_asteroids.csv
│   └── splits/                    # train.csv, val.csv, test.csv
├── results/                       # Saved metrics, plots, cross-validation results
├── notebooks/                     # Colab-compatible version
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/djasmine123/asteroid-threat-assessment.git
cd asteroid-threat-assessment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python src/asteroid_threat_model.py
```

The script will:
1. Fetch 150 real asteroids from NASA JPL API
2. Generate 5,000 physics-validated synthetic asteroids
3. Train the multimodal model with 5-fold cross-validation
4. Run ablation studies and baseline comparisons
5. Generate explainability reports (3 levels)
6. Save results to `results/`

### Run in Google Colab
The script is fully Colab-compatible. Upload `asteroid_threat_model.py` to Colab or open the notebook version in `notebooks/`.

---

## 📦 Dataset

**150 real asteroids** from [NASA JPL Close Approach Data API](https://ssd-api.jpl.nasa.gov/cad.api):
- Date range: 1995–2050
- Distance threshold: ≤ 0.3 AU (NASA PDCO standard)
- Features: absolute magnitude, distance, velocity, orbital elements

**5,000 synthetic asteroids** generated via physics-validated Monte Carlo sampling:
- H-G photometry system (Bowell et al., 1989)
- Keplerian orbital mechanics
- Spectral type distribution: S(40%), C(30%), X(15%), Q(10%), D(5%)
- KS-test confirms synthetic matches real: all p-values > 0.05

**Splits:** 70% train / 15% val / 15% test (stratified)

---

## 🧠 Key Techniques

**Cross-Modal Attention:** Learns which sensor is most reliable per asteroid — rather than treating all equally or using simple concatenation.

**Physics-Informed Loss:**
```
Total_Loss = CrossEntropy_Loss + 0.1 × Physics_Loss
Physics_Loss = MSE(D_predicted, D_HG) + MSE(T_predicted, T_Kepler)
```

**Three-Level Explainability:**
- **Engineers:** Sensor attribution percentages from attention weights
- **Scientists:** H-G and Kepler consistency scores
- **Decision-makers:** Natural language — "Very likely hazardous (89% confidence)"

---

## 📈 Ablation Study

| Configuration | Accuracy | ECE | H-G Consistency |
|---------------|----------|-----|-----------------|
| Full Model | **94.78%** | **0.0245** | **0.96** |
| No Physics Layers | 91.62% | 0.0389 | 0.67 |
| No Attention | 92.05% | 0.0312 | 0.94 |
| No Sequential Updates | 92.41% | 0.0278 | 0.96 |
| Basic Multimodal | 89.13% | 0.0521 | 0.58 |

---

## 🔬 Case Study: Apophis (99942)

The sequential update mechanism was validated on the well-known asteroid 99942 Apophis:

| Time Point | Sensors Available | Confidence | Primary Sensor |
|------------|------------------|------------|----------------|
| T1 | Optical only | 62% | Optical 100% |
| T2 | Optical + Radar | 78% | Radar 52%, Optical 48% |
| T3 | All sensors | 89% | Radar 42%, Optical 35%, Spectral 23% |

---

## 🛠️ Tech Stack

- **Python 3.8**, PyTorch 1.12
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- SciPy (KS-tests, physics validation)
- NASA JPL API (real data)

---

## 📄 Report

Full methodology, results, and analysis in `final_project_report.pdf` (SRIHER B.Sc. Data Science Final Project, 2026).

---

## 👩‍💻 Author

**Jasmine D**  
B.Sc. Data Science — Sri Ramachandra Institute of Higher Education and Research (SRIHER), Chennai  
GitHub: [@djasmine123](https://github.com/djasmine123) | LinkedIn: [djasmine1610](https://linkedin.com/in/djasmine1610)

---

## 📚 Key References

- Jean Jacques (2025) — GNN for hazardous asteroid classification. arXiv:2504.18605
- Abdullah et al. (2025) — XAI + anomaly detection for asteroid hazards. arXiv:2503.15901
- Khan et al. (2025) — Dual-balancing PINNs. IJCAI 2025
- Zhang et al. (2026) — Physics-informed GNN. Nature Communications 16:845
