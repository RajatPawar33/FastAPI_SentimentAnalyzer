import os
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib
from datetime import datetime

class SentimentModelTrainer:
    def __init__(self, data_path: str = "IMDB Dataset.csv", models_dir: str = "models"):
        self.data_path = data_path
        self.models_dir = models_dir
        self.vectorizer = None
        self.models = {}
        self.metrics = {}

        os.makedirs(self.models_dir, exist_ok=True)

        self.model_configs = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'naive_bayes': MultinomialNB(),
            'svm': LinearSVC(random_state=42, max_iter=10000)
        }

    def load_and_preprocess_data(self) -> Tuple[pd.Series, pd.Series]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)

        df.dropna(inplace=True)

        sentiment_mapping = {'positive': 1, 'negative': 0}
        df['sentiment'] = df['sentiment'].map(sentiment_mapping)
        df.dropna(subset=['sentiment'], inplace=True)

        return df['review'], df['sentiment']

    def create_vectorizer(self, X_train: pd.Series) -> TfidfVectorizer:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        vectorizer.fit(X_train)
        return vectorizer

    def train_models(self, X_train_vec, X_test_vec, y_train, y_test) -> Dict[str, Any]:
        results = {}
        for model_name, model in self.model_configs.items():
            try:
                model.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')

                results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }

                self.models[model_name] = model

                print(f"\n{model_name.upper()} Classification Report:\n")
                print(classification_report(y_test, y_pred))

            except Exception as e:
                print(f"Training failed for {model_name}: {e}")
        return results

    def save_models(self) -> None:
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(self.models_dir, f"{model_name}.pkl"))

        joblib.dump(self.vectorizer, os.path.join(self.models_dir, "vectorizer.pkl"))
        joblib.dump(self.metrics, os.path.join(self.models_dir, "metrics.pkl"))

        metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'vectorizer_features': len(self.vectorizer.vocabulary_),
            'metrics_summary': self.metrics
        }
        joblib.dump(metadata, os.path.join(self.models_dir, "training_metadata.pkl"))

    def print_model_comparison(self) -> None:
        if not self.metrics:
            print("No metrics to compare.")
            return

        print("\n" + "="*40)
        print("MODEL COMPARISON SUMMARY")
        print("="*40)

        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")

        best_model = max(self.metrics.items(), key=lambda x: x[1]['f1_score'])
        print(f"\nBest Model: {best_model[0].upper()} (F1 Score: {best_model[1]['f1_score']:.4f})")

    def run_training_pipeline(self) -> None:
        X, y = self.load_and_preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        self.vectorizer = self.create_vectorizer(X_train)

        X_train_vec = self.vectorizer.transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.metrics = self.train_models(X_train_vec, X_test_vec, y_train, y_test)

        self.save_models()

        # self.print_model_comparison()

def main():
    trainer = SentimentModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()
