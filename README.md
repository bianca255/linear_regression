# Student Math Performance Predictor

## Mission and Problem
Empower underprivileged children through technology and digital education so they can thrive in a technology-driven world.
Predict student final math outcomes (G3) early using learning and socio-demographic signals.
Enable schools and community programs to identify at-risk learners before final exams.
Support targeted, data-driven interventions that improve learning outcomes and long-term opportunity.

**Dataset:** [UCI Student Performance Dataset](https://www.kaggle.com/datasets/whenamancodes/student-performance) | 395 students x 33 features covering grades (G1, G2), studytime, absences, parental education, family support, and internet access. Target: G3 (final math grade, 0-20). Source: UCI Machine Learning Repository / Kaggle.

---

## Repository Structure

```text
linear_regression_model/
|
+-- summative/
	+-- linear_regression/
	|   +-- multivariate.ipynb
	+-- API/
	+-- FlutterApp/
```

## Run in Colab

https://colab.research.google.com/drive/1lgRKLVouRWqRasW3dkpiguc0nrrrfnW_#scrollTo=2tMt9GdRwNv6

---

## Live Deployment

**API Endpoint:** https://linear-regression-j8x8.onrender.com

**Swagger UI / API Documentation:** https://linear-regression-j8x8.onrender.com/docs

The FastAPI service is deployed on Render and provides three endpoints:
- `GET /` - Welcome message
- `GET /health` - Health check (returns model type)
- `POST /predict` - Make predictions (accepts 26-feature StudentInput)
- `POST /retrain` - Retrain model with new data

---

## How to Run

1. Download `student-mat.csv` from [Kaggle](https://www.kaggle.com/datasets/whenamancodes/student-performance) and place it alongside the notebook.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib
   ```
3. Open and run `multivariate.ipynb` top to bottom.

---

## Models Trained

| Model | Test MSE | Test R2 |
|-------|----------|---------|
| Linear Regression | 4.75 | 0.77 |
| Decision Tree | depth-limited, max_depth=6 | - |
| Random Forest | 200 estimators, max_depth=10 | - |

Best model (lowest Test MSE) saved automatically as `best_model.pkl`.

---

## Visualizations

- Correlation heatmap across all 33 encoded features
- Top feature importances bar chart (correlation with G3)
- G3 final grade distribution and boxplot
- Feature distributions (studytime, absences, G1, G2, Dalc, Walc)
- Scatter plots: G1, G2, studytime, absences vs G3 with trend lines
- Gradient Descent loss curve (Train vs Test MSE per epoch)
- Before/After scatter plot with fitted regression line (G2 vs G3)
- Model comparison bar charts (MSE and R2)

---

## Flutter Mobile App

**Setup:**
```bash
cd linear_regression_model/summative/FlutterApp
flutter pub get
flutter run
```

The Flutter app provides a user-friendly interface to make predictions:
- 26 input fields organized into 6 sections (Demographic, Parent Info, School, Support, Lifestyle, Academic)
- Form validation with min/max range enforcement
- Real-time predictions via HTTP POST to the live API
- Color-coded result interpretation (At risk / Passing / Good performance)

**API URL Configuration:**
The app is pre-configured to use: `https://linear-regression-j8x8.onrender.com/predict`

---

