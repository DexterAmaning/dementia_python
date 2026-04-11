# dementia_python
confusion matrix
Dementia Detection — Multiclass Cognitive State Classification
SVM-based screening tool using MRI-derived biomarkers and clinical features
![Python](https://img.shields.io/badge/Python-3.8+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-orange) ![Domain](https://img.shields.io/badge/Domain-Neurology-purple) ![Pipeline](https://img.shields.io/badge/Pipeline-Production--level-blue) ![Status](https://img.shields.io/badge/Status-Complete-green)


Overview
This project builds a Support Vector Machine (SVM) model to classify cognitive states — Demented, Nondemented, and Converted — using MRI-derived biomarkers and structured clinical features. The system simulates a neurodegenerative disease screening tool for early detection and longitudinal monitoring of dementia progression.
> The hardest clinical challenge in dementia is not diagnosing clear disease states —  
> it is detecting the **transition**: patients converting from healthy to demented.  
> That is where this model focuses its improvement effort.


Clinical Motivation
Challenge	Relevance
Late-stage diagnosis	Dementia is often identified only after significant cognitive decline
MRI underutilization	Structural brain changes are measurable years before symptoms present
Global aging burden	Dementia prevalence is rising — scalable screening tools are critical
Progression monitoring	Identifying "Converted" patients early enables timely intervention
Quality of life impact	Earlier detection allows better care planning and slower progression


Dataset
Property	Value
Source	OASIS Longitudinal MRI Dataset
Samples	373 longitudinal MRI records
Features	MRI-derived biomarkers + clinical and demographic variables
Target	3-class: `Nondemented`, `Demented`, `Converted`
Missing values	Present in `SES` and `MMSE` — requires imputation
Feature Overview
Feature	Type	Clinical Meaning
`MMSE`	Cognitive	Mini-Mental State Examination score — primary dementia screen
`CDR`	Clinical	Clinical Dementia Rating — gold standard severity scale (0–3)
`eTIV`	MRI-derived	Estimated total intracranial volume — brain size normalization
`nWBV`	MRI-derived	Normalized whole brain volume — atrophy indicator
`ASF`	MRI-derived	Atlas scaling factor — volumetric correction
`Age`	Demographic	Risk increases sharply after 65
`EDUC`	Demographic	Years of education — cognitive reserve proxy
`SES`	Demographic	Socioeconomic status — social determinant of health
`M/F`	Demographic	Sex — affects dementia prevalence and presentation
> **Data quality note:** Missing values in `SES` and `MMSE` were handled via median imputation. `MMSE` missingness in dementia data is clinically meaningful — patients with severe impairment often cannot complete the test — and should be modeled carefully.



ML Pipeline
```
Longitudinal MRI Records (373 samples)
        ↓
Missing Value Imputation (SES, MMSE — median)
        ↓
Label Encoding (categorical features)
        ↓
Feature Standardization (StandardScaler)
        ↓
SMOTE — Minority Class Oversampling (Converted class)
        ↓
Train / Test Split (stratified)
        ↓
SVM Training + GridSearchCV Hyperparameter Tuning
        ↓
Evaluation (Multiclass: Precision, Recall, F1 per class)
        ↓
Classification: Nondemented / Demented / Converted
```
> This pipeline implements production-level ML practices: imputation, encoding, scaling, class balancing, and hyperparameter optimization — not just a baseline fit.


Model & Techniques
Why SVM?
Effective in high-dimensional feature spaces with small sample sizes
Handles nonlinear decision boundaries via RBF kernel
Strong generalization when combined with proper scaling and tuning
Well-validated in neuroimaging and clinical classification literature
Advanced Techniques Applied
Technique	Purpose
`Label Encoding`	Convert categorical sex variable to numeric
`Median Imputation`	Handle missing SES and MMSE values
`StandardScaler`	Normalize MRI and clinical features to equal scale
`SMOTE`	Oversample underrepresented `Converted` class
`GridSearchCV`	Systematic hyperparameter tuning (C, gamma, kernel)



Results
Final Model Performance (Tuned SVM)
Overall Accuracy: ~89–91%
Classification Report
Class	Precision	Recall	F1-score	Support
Converted	0.40	0.22	0.29	9
Demented	0.95	0.95	0.95	37
Nondemented	0.90	0.98	0.94	48
Confusion Matrix
```
                 Predicted:       Predicted:     Predicted:
                 Converted        Demented        Nondemented
Actual: Converted     2               2               5
Actual: Demented      2              35               0
Actual: Nondemented   1               0              47
```
Key Insight
The model performs exceptionally well on clear disease states:
Demented: 95% precision and recall — near-perfect detection
Nondemented: 98% recall — very few healthy patients misclassified
The challenge is the Converted class (22% recall):
Only 2 out of 9 converting patients correctly identified
5 were misclassified as Nondemented — the highest-cost error
> This reflects a genuine clinical reality: the transition from healthy to demented is the hardest cognitive state to detect. Subtle MRI changes and borderline MMSE scores make this a fundamentally difficult classification problem — not a model failure.


Limitations
Limitation	Impact
Small dataset (373 samples)	High overfitting risk — especially for minority Converted class
Class imbalance (few Converted)	SMOTE helps but cannot fully compensate for 9-sample minority
Longitudinal structure underutilized	Time between visits not modeled — temporal progression ignored
Static snapshot classification	Each record treated independently — not as a patient trajectory
MMSE imputation in severe cases	Missing MMSE may indicate severity — median fill loses this signal
No external validation	Model not tested on non-OASIS dementia populations


Planned Improvements
[ ] LSTM / temporal modeling — leverage longitudinal visit sequences to track progression over time
[ ] Integrate MRI image data — add CNN-extracted spatial features alongside structured biomarkers
[ ] Improve Converted class recall — cost-sensitive learning with higher misclassification penalty for conversion cases
[ ] SHAP explainability — visualize which MRI and clinical features drive each patient's classification
[ ] Patient-level trajectory analysis — group records by subject ID and model transition probability
[ ] Threshold optimization — lower decision boundary for Converted class to reduce false negatives


Future Direction
Build a longitudinal risk trajectory dashboard — visualize a patient's cognitive state across visits
Combine MRI volumetric features with cognitive test time-series for richer signal
Deploy as a Streamlit screening tool for neuropsychology clinic use
Validate on external dementia cohorts (ADNI, UK Biobank)
Evaluate performance across age, sex, and education subgroups for fairness


What This Project Demonstrates
Multiclass classification in a real-world clinical setting
Handling genuinely messy medical data — missing values, class imbalance, longitudinal structure
Production-level ML pipeline: imputation → encoding → scaling → SMOTE → tuning
Clinical nuance in performance interpretation — understanding why Converted is hard
Pathway toward time-series and imaging-based extensions


Tech Stack
`Python` `scikit-learn` `pandas` `NumPy` `imbalanced-learn (SMOTE)` `Matplotlib` `Seaborn`


Related Projects
Project	Description
Heart Disease Prediction	Logistic Regression cardiovascular risk classifier
Diabetes Prediction Model	SVM classifier for Type 2 diabetes risk
ASD Detection Model	Behavioral screening ML for Autism Spectrum Disorder
Parkinson's Detection Model	Voice biomarker SVM for Parkinson's Disease
Medicaid Drug Spending Dashboard	Power BI forecasting for Medicaid drug utilization


References
Marcus DS et al. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle-aged, nondemented, and demented older adults. Journal of Cognitive Neuroscience.
Folstein MF et al. (1975). Mini-mental state: A practical method for grading the cognitive state of patients for the clinician. Journal of Psychiatric Research.
OASIS Brains Project — oasis-brains.org


Author
Prince Kwarteng Amaning  
MS Data Science (in progress) — University of Michigan–Dearborn  
Background: Dentistry | Global Public Health | Healthcare Analytics
Interested in building AI systems for neurodegenerative disease detection — where early identification of disease progression can meaningfully change patient outcomes.


License
MIT License — open for educational and research use.
