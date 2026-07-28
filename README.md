# 🎮 FPS Benchmark Prediction using Machine Learning

A Machine Learning project that predicts **Frames Per Second (FPS)** in PC games based on hardware specifications such as CPU, GPU, memory bandwidth, clock speed, VRAM, and other benchmark features.

The project analyzes gaming benchmark data, performs feature engineering, compares multiple regression algorithms, and identifies the most accurate model for FPS prediction.

This project was developed as part of the **Bachelor of Engineering in Robotics and Artificial Intelligence** at **Thapar Institute of Engineering and Technology**.

---

# 📌 Project Overview

Gaming performance depends heavily on computer hardware. Choosing the right hardware configuration can significantly improve gameplay while reducing unnecessary upgrade costs.

This project uses Machine Learning to learn the relationship between hardware specifications and gaming FPS, allowing performance prediction for unseen hardware configurations.

---

# 🎯 Objectives

- Collect and analyze FPS benchmark data
- Clean and preprocess gaming hardware data
- Perform feature engineering
- Compare multiple regression algorithms
- Predict FPS using machine learning
- Identify hardware components with the highest impact on gaming performance

---

# ✨ Features

- 🎮 FPS prediction using hardware specifications
- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data preprocessing and cleaning
- 🔍 Correlation analysis
- ⚙ Feature engineering
- 🤖 Multiple regression model comparison
- 📈 XGBoost implementation
- 📉 Model performance evaluation using R² score

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

# 📂 Dataset

The project uses an FPS benchmark dataset containing gaming hardware specifications including:

- CPU information
- GPU information
- GPU Clock Speed
- VRAM Size
- Memory Bandwidth
- Core Count
- Pixel Rate
- Texture Fill Rate
- Thermal Design Power
- FPS values

---

# ⚙ Data Preprocessing

The dataset was preprocessed by:

- Removing unnecessary characters
- Handling missing values
- Feature selection
- Feature engineering
- One-hot encoding of categorical variables
- Correlation analysis

---

# 🤖 Machine Learning Models

The following regression models were implemented:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The models were trained and evaluated using the R² metric.

---

# 🏆 Best Performing Model

**XGBoost Regressor**

Performance:

- R² Score: **0.99**
- Features Used: **15**
- High prediction accuracy
- Excellent handling of non-linear relationships

---

# 📈 Key Findings

- GPU specifications have the strongest influence on gaming FPS.
- Machine Learning can accurately estimate FPS from hardware specifications.
- XGBoost achieved the highest prediction accuracy among all evaluated models.
- Different games exhibit varying FPS behavior under identical hardware configurations.

---

# 📁 Project Structure

FPS-Benchmark-Prediction/

├── fps_prediction.py

├── main.ipynb

├── fps_benchmark.csv

├── requirements.txt

├── README.md

└── .gitignore

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/FPS-Benchmark-Prediction.git
```

Move into the project directory

```bash
cd FPS-Benchmark-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

Launch the notebook

```bash
jupyter notebook
```

or execute

```bash
python fps_prediction.py
```

---

# 📊 Results

Model Comparison

| Model | Performance |
|--------|-------------|
| Linear Regression | Good |
| Random Forest | Very Good |
| **XGBoost** | **Best (R² = 0.99)** |

---

# 🚀 Future Improvements

- Deep Learning-based FPS prediction
- Real-time FPS prediction application
- Larger benchmark datasets
- Multi-resolution prediction
- Cloud deployment
- Web dashboard for predictions

---

# 👩‍💻 Author

**Shreya**

B.E. Robotics & Artificial Intelligence

Thapar Institute of Engineering and Technology
