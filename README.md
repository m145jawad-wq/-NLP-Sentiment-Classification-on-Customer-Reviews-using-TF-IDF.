NLP Sentiment Classification Using TF-IDF
A machine learning project that uses Natural Language Processing (NLP) and TF-IDF to classify customer reviews as positive or negative using Multinomial Naive Bayes and Logistic Regression.

📌 Project Overview
Sentiment analysis identifies the emotional tone of text. This project processes customer, movie, and restaurant reviews to predict whether a review expresses positive or negative sentiment.

Key Features
Text cleaning and preprocessing
TF-IDF feature extraction
Multinomial Naive Bayes classification
Logistic Regression classification
Model accuracy comparison
Confusion matrix visualizations
Precision, recall, and F1-score evaluation
Model saving and loading with Joblib
Interactive sentiment prediction
🛠️ Technologies Used
Python
Pandas
Scikit-learn
NLTK
Matplotlib
Seaborn
Joblib
📂 Project Structure

text
NLP-Sentiment-Classification/
├── sentiment_classification.py
├── README.md
├── sentiment_data/
│   ├── sentiment_dataset.zip
│   ├── processed_reviews.csv
│   └── model_comparison.csv
└── saved_models/
    ├── best_sentiment_model.joblib
    ├── tfidf_vectorizer.joblib
    └── metadata.joblib
The sentiment_data/ and saved_models/ folders are generated when the project runs.

📊 Dataset
This project uses the UCI Sentiment Labelled Sentences dataset, containing reviews from:

Amazon product reviews
IMDb movie reviews
Yelp restaurant reviews
Dataset source: UCI Machine Learning Repository

Labels:

0 — Negative sentiment
1 — Positive sentiment
⚙️ Installation and Setup
1. Clone the repository

bash
git clone https://github.com/YOUR_USERNAME/NLP-Sentiment-Classification.git
cd NLP-Sentiment-Classification
Replace YOUR_USERNAME with your GitHub username.

2. Install dependencies

bash
pip install pandas scikit-learn nltk matplotlib seaborn joblib
3. Train the models
Run the following command to download the dataset, preprocess the text, train both classifiers, evaluate their performance, and save the best model.


bash
python sentiment_classification.py --train
An internet connection is required for the initial dataset download.

4. Predict sentiment
After training, run:


bash
python sentiment_classification.py --predict
Enter a customer review when prompted. Type exit to stop.

🔄 Workflow
Download the labeled review dataset.
Clean and preprocess review text.
Convert text into numerical features using TF-IDF.
Split the dataset into 80% training and 20% testing data.
Train Multinomial Naive Bayes and Logistic Regression.
Evaluate both models on the test set.
Select the model with the highest test accuracy.
Save the trained model and TF-IDF vectorizer.
Load the saved files to predict the sentiment of new reviews.
🤖 Machine Learning Algorithms
Multinomial Naive Bayes
A probabilistic algorithm commonly used for text classification. It is computationally efficient and provides a useful baseline.

Logistic Regression
A linear classification algorithm that learns patterns in TF-IDF features to distinguish positive and negative reviews.

📈 Evaluation Metrics
The project evaluates the classifiers using:


Metric	Description
Accuracy	Overall proportion of correct predictions
Precision	Correct predictions among predicted examples of a class
Recall	Correctly identified examples among actual examples of a class
F1-score	Harmonic mean of precision and recall
Confusion Matrix	Summary of correct and incorrect classifications
The script also produces confusion matrix plots for both classifiers.

Actual scores are calculated when you run the project; no specific accuracy is guaranteed.

💾 Model Saving and Loading
The best-performing classifier and its matching TF-IDF vectorizer are saved using Joblib.


File	Purpose
best_sentiment_model.joblib	Trained classifier
tfidf_vectorizer.joblib	Fitted TF-IDF vectorizer
metadata.joblib	Model configuration and name
Saved artifacts allow future predictions without retraining the models.

Security note: Only load Joblib files from trusted sources.

💬 Example
Input:


text
This product is excellent and works perfectly!
Expected sentiment:


text
Positive
Input:


text
The service was disappointing and the food was cold.
Expected sentiment:


text
Negative
These are illustrative examples; actual predictions depend on the trained model.

🚀 Future Improvements
Build an interactive Streamlit web application.
Add neutral sentiment classification.
Perform hyperparameter tuning and cross-validation.
Experiment with additional NLP techniques.
Deploy the classifier as a REST API.
Add automated tests and model versioning.
🎓 Learning Outcomes
This project demonstrates how to:

Apply NLP preprocessing techniques.
Extract TF-IDF features from text.
Train and compare machine learning models.
Evaluate classification performance.
Visualize model results.
Save and reuse trained models with Joblib.
👨‍💻 Author
M JAWAD MAZHAR

GitHub: M JAWAD


📄 License
This project is intended for educational purposes. Consider adding an open-source license, such as the MIT License, before distributing your code. Review the dataset's terms before redistributing its contents.

⭐ If you find this project useful, consider giving the repository a star
