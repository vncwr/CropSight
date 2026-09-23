# CropSight: Edge-AI Crop Disease Diagnosis & Agronomic Forensic Investigation Platform

<p align="center">
  <img src="assets/icons/cropsight_f.png" alt="CropSight Logo" width="160" />
</p>

<p align="center">
  <strong>On-Device Deep Learning Inference • Forensic Contamination Tracing • Agronomic Intelligence</strong>
</p>

---

## Pre-Built Production Application (Android APK)

> [!NOTE]
> The mobile client source code is retained in private archives for intellectual property and research integrity. The complete, production-ready Android application package is distributed via GitHub Releases.
> 
> **[Download CropSight-v1.apk from GitHub Releases](../../releases/latest)**

---

## Academic Compliance & Research Credits

This research project and engineering methodology was conceptualized, designed, and developed in compliance with the academic requirements for the subject **Application Development**.

### Development & Research Team
- **Vince Warren Pradas** — *Lead Architect & Lead Developer*
- **Mark Niel Mahinay** — *Core Team Member / Systems Research*
- **Arbe Maquiling** — *Core Team Member / Agronomic & Data Research*

### Mission Statement
> *"Dedicated to empowering smallholder farmers, defending regional food sovereignty, and equipping agricultural field workers with forensic-grade diagnostic intelligence to intercept pathogen outbreaks before catastrophic crop failure."*

---

## Scientific & Methodological Framework

CropSight is an edge-AI crop disease diagnosis and agronomic forensic investigation platform engineered to operate in remote agricultural environments under computational and connectivity constraints. 

While the platform architecture is designed as a generalized framework for multi-crop phytopathological diagnosis, empirical baseline training and validation were initiated using rice crop (*Oryza sativa*) foliar pathogens as the foundational benchmark. This initial focus addresses high-impact staple grain pathogens while providing a modular transfer-learning pipeline for ongoing expansion into additional cereal, legume, and horticultural crops.

```
                         [ Field Crop Foliar Sample ]
                     (Baseline Benchmark: Oryza sativa)
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │    On-Device MobileNetV2 Architecture    │
                │  - Depthwise Separable Convolutions      │
                │  - 224x224 RGB Quantized Float32 Model   │
                │  - Sub-100ms Offline Field Inference     │
                └────────────────────┬────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              ▼                                             ▼
 [ Pathogen Classification ]                  [ Forensic Geospatial Engine ]
 • Bacterial Leaf Blight                      • GPS Coordinates (Lat/Lon)
 • Brown Spot                                 • Wind Vector Plume Dispersion
 • Leaf Smut                                  • OSM Overpass Waterway Risk Scan
 (Modular for Multi-Crop Expansion)                         │
              │                                             │
              └──────────────────────┬──────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │      Serverless Agronomist Proxy        │
                │   - Ephemeral Vercel Cloud Architecture │
                │   - Secret-Isolated Groq Llama 3.3 70B  │
                │   - Context-Aware Remediation & Chat    │
                └─────────────────────────────────────────┘
```

---

### 1. On-Device Edge Deep Learning
The diagnostic engine leverages a fine-tuned **MobileNetV2** convolutional neural network optimized for low-latency, on-device inference without cellular dependency.

- **Tensor Input**: $224 \times 224 \times 3$ RGB normalized tensor.
- **Model Architecture**: Inverted residual blocks with linear bottlenecks, pre-trained on ImageNet and specialized via supervised transfer learning.
- **Runtime Footprint**: TensorFlow Lite format (`assets/model/model.tflite`, ~2.8 MB), engineered for minimal memory overhead and consistent sub-100ms execution on standard mobile hardware.
- **Initial Pathogen Target Classes (Rice Benchmark)**:
  1. **Bacterial Leaf Blight** (*Xanthomonas oryzae pv. oryzae*)
  2. **Brown Spot** (*Bipolaris oryzae*)
  3. **Leaf Smut** (*Entyloma oryzae*)

### 2. Forensic Contamination & Atmospheric Dispersion Modeling
CropSight couples visual classification with spatial and atmospheric telemetry:
- **Atmospheric Plume Modeling**: Evaluates directional wind vectors and velocity ($m/s$) to project potential airborne fungal spore and bacterial aerosol spread corridors.
- **Hydrological Vector Mapping**: Interfaces with the OpenStreetMap Overpass API to analyze adjacent irrigation canals, drainage ditches, and waterways that facilitate the transmission of waterborne pathogens.

### 3. Serverless Agronomic Intelligence Proxy
To maintain rigorous security and prevent client-side credential exposure, the mobile client communicates through an ephemeral serverless proxy hosted on Vercel (`backend/`):
- Isolates API secrets from edge distribution packages.
- Orchestrates high-throughput agronomic synthesis via Groq Llama 3.3 70B inference endpoints.
- Incorporates an offline agronomic lookup catalog (`assets/treatments/treatment_lookup.json`) ensuring uninterrupted operational capability in remote rural zones.

---

## Repository Structure

```
CropSight-Research/
├── assets/
│   ├── icons/                  # Application branding and logos
│   │   └── cropsight_f.png
│   ├── model/                  # Compiled TensorFlow Lite deployment model & labels
│   │   ├── labels.txt          # Target diagnostic classes
│   │   └── model.tflite        # Optimized edge-inference neural network (2.8 MB)
│   └── treatments/             # Curated agronomic disease treatment lookup database
│       └── treatment_lookup.json
├── backend/                    # Serverless Cloud Proxy Architecture
│   ├── api/                    # Vercel Serverless Function entrypoints
│   │   ├── index.js            # Health check and endpoint routing
│   │   └── summarize.js        # Groq Llama 3.3 70B LLM orchestration
│   ├── package.json            # Node.js dependencies
│   └── vercel.json             # Deployment routing specifications
├── ml/                         # Machine Learning Pipeline & Experiments
│   ├── scripts/
│   │   ├── 01_explore_dataset.py   # Dataset distribution and health audit
│   │   ├── 02_prepare_splits.py    # Stratified train/val/test partitioning
│   │   ├── 03_train.py             # Transfer learning training loop
│   │   ├── 04_evaluate.py          # Confusion matrix and F1-score evaluation
│   │   └── 05_convert_tflite.py    # TFLite conversion and optimization
│   ├── CropSight_Training.ipynb    # End-to-end Jupyter / Google Colab GPU notebook
│   ├── data_card.md                # Dataset lineage, evaluation metrics, data ethics
│   └── requirements.txt            # Python dependencies (TensorFlow, Scikit-learn, etc.)
├── .gitignore                  # Git exclusions for models, secrets, and datasets
├── LICENSE                     # MIT License
└── README.md                   # Research documentation and system specifications
```

---

## Machine Learning Pipeline Execution

### Environment Setup
```bash
cd ml
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Reproducing Baseline Model Training
The end-to-end pipeline is available in an interactive notebook compatible with Google Colab GPU runtimes:
```bash
jupyter notebook CropSight_Training.ipynb
```

Alternatively, run the modular pipeline sequentially:
```bash
python scripts/01_explore_dataset.py
python scripts/02_prepare_splits.py
python scripts/03_train.py
python scripts/04_evaluate.py
python scripts/05_convert_tflite.py
```

---

## License

This research archive, model architecture, and cloud proxy are distributed under the [MIT License](LICENSE).
