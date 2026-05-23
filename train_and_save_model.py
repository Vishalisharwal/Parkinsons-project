import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class ParkinsonsPipeline:
    """Complete ML pipeline for Parkinson's disease detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def load_and_prepare_data(self, filepath="parkinsons.csv"):
        """Load and prepare the dataset"""
        print("📊 Loading dataset...")
        df = pd.read_csv(filepath)
        
        # Separate features and target
        X = df.drop(['name', 'status'], axis=1)
        y = df['status']
        
        self.feature_names = X.columns.tolist()
        print(f"✅ Dataset loaded: {len(df)} samples, {len(self.feature_names)} features")
        
        return X, y
    
    def train_and_evaluate(self, X, y, test_size=0.2):
        """Train both models and compare performance"""
        print("\n🔄 Splitting and scaling data...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples\n")
        
        results = {}
        best_score = 0
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"🤖 Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            print(f"  ✓ Accuracy: {accuracy:.4f}")
            print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
            print(f"  ✓ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
            
            # Track best model
            if accuracy > best_score:
                best_score = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print(f"🏆 Best Model: {self.best_model_name} (Accuracy: {best_score:.4f})\n")
        
        # Display detailed results for best model
        self._display_best_model_results(results[self.best_model_name])
        
        return results
    
    def _display_best_model_results(self, result):
        """Display detailed classification report for best model"""
        print(f"📈 Detailed Results for {self.best_model_name}:")
        print("\nClassification Report:")
        print(classification_report(result['y_test'], result['y_pred'], 
                                   target_names=['Healthy', 'Parkinson\'s']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        print(f"  True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    def save_pipeline(self, model_path="best_model.pkl", scaler_path="scaler.pkl"):
        """Save the best model and scaler"""
        if self.best_model is None:
            print("❌ No model trained yet!")
            return
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, "feature_names.pkl")
        
        print(f"\n💾 Pipeline saved:")
        print(f"  - Model: {model_path}")
        print(f"  - Scaler: {scaler_path}")
        print(f"  - Features: feature_names.pkl")
    
    def predict(self, features):
        """
        Predict Parkinson's likelihood for new speech features
        
        Parameters:
        -----------
        features : array-like or dict
            Speech features (either as array in correct order or dict with feature names)
        
        Returns:
        --------
        dict with prediction results
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Handle dict input
        if isinstance(features, dict):
            features = [features.get(name, 0) for name in self.feature_names]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features_array)
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)[0]
        probability = self.best_model.predict_proba(features_scaled)[0]
        
        result = {
            'has_parkinsons': bool(prediction),
            'probability_healthy': probability[0],
            'probability_parkinsons': probability[1],
            'confidence': max(probability),
            'model_used': self.best_model_name
        }
        
        return result
    
    @staticmethod
    def load_pipeline(model_path="best_model.pkl", scaler_path="scaler.pkl"):
        """Load a saved pipeline"""
        pipeline = ParkinsonsPipeline()
        pipeline.best_model = joblib.load(model_path)
        pipeline.scaler = joblib.load(scaler_path)
        pipeline.feature_names = joblib.load("feature_names.pkl")
        pipeline.best_model_name = type(pipeline.best_model).__name__
        return pipeline


# ============ USAGE EXAMPLE ============

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ParkinsonsPipeline()
    
    # Load and prepare data
    X, y = pipeline.load_and_prepare_data("parkinsons.csv")
    
    # Train and evaluate models
    results = pipeline.train_and_evaluate(X, y)
    
    # Save the best model
    pipeline.save_pipeline()
    
    print("\n" + "="*60)
    print("🎯 PREDICTION EXAMPLE")
    print("="*60)
    
    # Example prediction using first sample from dataset
    sample_features = X.iloc[0].values
    prediction_result = pipeline.predict(sample_features)
    
    print(f"\n📊 Prediction Results:")
    print(f"  Status: {'Parkinson\'s Detected' if prediction_result['has_parkinsons'] else 'Healthy'}")
    print(f"  Probability of Parkinson's: {prediction_result['probability_parkinsons']:.2%}")
    print(f"  Probability of Healthy: {prediction_result['probability_healthy']:.2%}")
    print(f"  Confidence: {prediction_result['confidence']:.2%}")
    print(f"  Model Used: {prediction_result['model_used']}")
    
    print("\n" + "="*60)
    print("✅ Pipeline ready for deployment!")
    print("="*60)
    
    # Show how to use saved model
    print("\n💡 To use the saved model later:")
    print("   pipeline = ParkinsonsPipeline.load_pipeline()")
    print("   result = pipeline.predict(your_features)")