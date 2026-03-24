import os 
import joblib 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
def main(): 
    os.makedirs("models", exist_ok=True) 
    file_path = "data/train.csv" 
    df = pd.read_csv(file_path) 
    print("First 5 rows:") 
    print(df.head()) 
 
    print("\nDataset shape:", df.shape) 
    print("\nColumns:") 
    print(df.columns.tolist()) 
 
    selected_columns = [ 
        "Survived", "Pclass", "Sex", "Age", 
        "SibSp", "Parch", "Fare", "Embarked" 
    ] 
    df = df[selected_columns] 
 
    print("\nMissing values before handling:") 
    print(df.isnull().sum()) 
 
    median_age = df["Age"].median() 
    df["Age"] = df["Age"].fillna(median_age) 
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0]) 
 
    X = df.drop("Survived", axis=1) 
    y = df["Survived"] 
 
    X = pd.get_dummies(X, drop_first=True) 
 
    feature_columns = X.columns.tolist() 
    joblib.dump(feature_columns, "models/feature_columns.pkl") 
    joblib.dump(median_age, "models/median_age.pkl") 
 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, random_state=42, stratify=y 
    ) 
 
    model = DecisionTreeClassifier(max_depth=5, random_state=42) 
    model.fit(X_train, y_train)  
 
    y_pred = model.predict(X_test) 
 
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}") 
    print("\nConfusion Matrix:") 
    print(confusion_matrix(y_test, y_pred)) 
    print("\nClassification Report:") 
    print(classification_report(y_test, y_pred)) 
 
    joblib.dump(model, "models/decision_tree_model.pkl") 
 
    print("\nModel saved as models/decision_tree_model.pkl") 
    print("Feature columns saved as models/feature_columns.pkl") 
    print("Median age saved as models/median_age.pkl") 
 
 
if __name__ == "__main__": 
    main()
