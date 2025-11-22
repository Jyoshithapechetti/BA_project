Title: Uncovering the Hidden Risk A Data Analytics Framework for Patient-Reported Drug Reactions

Project Summary
This project performs end-to-end analytics on real-world drug review data, integrating Exploratory Data Analysis (EDA), NLP-based feature engineering, severity extraction, side-effect co-occurrence analysis, drug similarity modeling, and machine-learning–based sentiment and rating prediction.

The goal of the project is to:
⦁	Understand patient experiences with different drugs
⦁	Identify common side effects and their severities
⦁	Analyse similarity relationships between drugs
⦁	Build predictive models to classify sentiment and predict user-assigned ratings
⦁	Provide decision-support insights for healthcare stakeholders, researchers, and pharmaceutical companies

The system is deployed through an interactive Streamlit dashboard and includes an integrated ML prediction module for real-time review analysis.

Team Information
Student Name:
Jyoshitha Pechetti
Course: Business Analytics 
Team no: 16

Key Features
⦁	Data cleaning, preprocessing, and review normalization
⦁	Token and character length analytics
⦁	Side-effect extraction using a medical dictionary and phrase-based patterns
⦁	Severity detection using medical keywords, multiword patterns, and negation handling
⦁	Side-effect co-occurrence analysis using PMI and Jaccard similarity
⦁	Drug–drug similarity analysis using cosine similarity on normalized side-effect frequencies
⦁	Risk-level classification for each drug
⦁	Machine Learning module for:
	Sentiment classification (Positive vs Negative)
	Rating prediction (1 to 10 scale)
⦁	Interactive Streamlit dashboard with multiple analytical views
⦁	Predictive box for entering any new review text

Technology Stack
⦁	Python
⦁	Streamlit
⦁	Pandas
⦁	Scikit-learn
⦁	Matplotlib
⦁	VADER Sentiment
⦁	NumPy
⦁	TF-IDF NLP pipelines
⦁	Logistic Regression, Ridge Regression
⦁	Co-occurrence and similarity algorithms (PMI, Jaccard, Cosine)

How to Run
1.	Install dependencies
2.	Place dataset files (drugsComTrain_raw.csv, drugsComTest_raw.csv) inside data/
3.	Run the Streamlit app
command to run: streamlit run app.py

Additional Resources
⦁	Scikit-learn Documentation
⦁	Streamlit API Docs
⦁	NLP TF-IDF Feature Extraction Guides
⦁	Research papers on adverse-event mining and pharmacovigilance
