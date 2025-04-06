import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def read_data():
    _column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                     'marital-status', 'occupation', 'relationship', 'race', 'sex',
                     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    _train_data = pd.read_csv("adult/adult.data", names=_column_names, sep=', ', na_values=' ?', engine='python')

    _test_data = pd.read_csv("adult/adult.test", names=_column_names, sep=', ', na_values=' ?',
                             engine='python')
    return _train_data, _test_data


def get_pipeline():
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
    numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                          'capital-loss', 'hours-per-week']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    svc_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, C=1.0, kernel='rbf', gamma='scale'))
    ])

    return svc_pipeline


def create_model(_pipeline, _train_data, _test_data):
    _test_data['income'] = _test_data['income'].str.replace('.', '')

    _test_data['income'] = _test_data['income'].str.replace('.', '')
    _test_data['income'] = _test_data['income'].str.strip()
    _train_data['income'] = _train_data['income'].str.strip()

    _train_data = _train_data.replace('?', np.nan)
    _test_data = _test_data.replace('?', np.nan)

    _train_data.dropna(inplace=True)
    _test_data.dropna(inplace=True)

    _train_data["income"] = _train_data["income"].map({">50K": 1, "<=50K": 0})
    _test_data["income"] = _test_data["income"].map({">50K": 1, "<=50K": 0})

    print("\nMissing values in training data:")
    print(_train_data.isnull().sum())
    print("\nMissing values in test data:")
    print(_test_data.isnull().sum())

    X_train = _train_data.drop("income", axis=1)
    y_train = _train_data["income"]
    X_test = _test_data.drop("income", axis=1)
    y_test = _test_data["income"]

    _pipeline.fit(X_train, y_train)
    y_pred = _pipeline.predict(X_test)

    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<=50K', '>50K'],
                yticklabels=['<=50K', '>50K'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print("Confusion matrix saved to 'confusion_matrix.png'")


def predict_income(model, age, workclass, education, marital_status, occupation,
                   relationship, race, sex, capital_gain, capital_loss,
                   hours_per_week, native_country):
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [0],
        'education': [education],
        'education-num': [0],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    education_mapping = {
        ' Preschool': 1,
        ' 1st-4th': 2,
        ' 5th-6th': 3,
        ' 7th-8th': 4,
        ' 9th': 5,
        ' 10th': 6,
        ' 11th': 7,
        ' 12th': 8,
        ' HS-grad': 9,
        ' Some-college': 10,
        ' Assoc-voc': 11,
        ' Assoc-acdm': 12,
        ' Bachelors': 13,
        ' Masters': 14,
        ' Prof-school': 15,
        ' Doctorate': 16
    }

    if education in education_mapping:
        input_data['education-num'] = education_mapping[education]

    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    print(f"Prediction: {'Income >$50K' if prediction == 1 else 'Income ≤$50K'}")
    print(f"Probability of income >$50K: {probability:.2f}")

    return prediction, probability


print('-----------------------------')

train_data, test_data = read_data()
pipeline = get_pipeline()
create_model(pipeline, train_data, test_data)

predict_income(pipeline,
               age=45,
               workclass=' Private',
               education=' Bachelors',
               marital_status=' Married-civ-spouse',
               occupation=' Exec-managerial',
               relationship=' Husband',
               race=' White',
               sex=' Male',
               capital_gain=0,
               capital_loss=0,
               hours_per_week=45,
               native_country=' United-States')
