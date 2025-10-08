# Fake Job Detector using NLP

## About the Project

This project implements a Fake Job Detection system using Natural Language Processing (NLP) and Machine Learning techniques. The system is designed to analyze job postings and determine whether they are genuine or fraudulent, helping job seekers avoid scams and illegitimate job offers.

### Technical Implementation

#### NLP Components
- **Text Preprocessing**: 
  - Utilizes NLTK (Natural Language Toolkit) for text processing
  - Implements stopword removal to filter out common words
  - Applies lemmatization using WordNetLemmatizer to reduce words to their base form
  - Includes custom text cleaning with regular expressions

#### Feature Engineering
- **TF-IDF Vectorization**: 
  - Transforms job descriptions into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)
  - Captures the importance of words in job postings while accounting for their frequency across all postings
  - Creates a rich feature space for the machine learning model

#### Machine Learning
- **Model**: Uses Logistic Regression for classification
- **Binary Classification**: Predicts whether a job posting is legitimate or fake
- **Serialized Models**: Saves and loads trained models using joblib for efficient deployment

### Features
- Interactive web interface built with Streamlit
- Real-time job posting analysis
- User feedback collection system for continuous improvement
- Responsive UI with modern design elements
- Background blur effects and professional styling

### Dataset
The system uses a comprehensive dataset of job postings (`fake_job_postings.csv`) containing both legitimate and fraudulent job listings.

### How to Use
1. The application can be accessed through a web interface
2. Input a job description in the text area
3. Click submit to get an analysis of whether the job posting is likely to be fake or genuine
4. Provide feedback to help improve the system's accuracy

### Technical Requirements
- Python
- Streamlit
- NLTK
- scikit-learn
- pandas
- joblib

### Project Structure
```
├── app.py              # Main application file with Streamlit interface
├── data/
│   ├── fake_job_postings.csv    # Training dataset
│   └── feedback_data.csv        # User feedback collection
├── model/
│   ├── fake_job_model.pkl       # Trained ML model
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
└── requirements.txt    # Project dependencies
```
