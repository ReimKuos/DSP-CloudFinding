import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib
import geopandas as gpd
import random
from shapely.geometry import Point

def random_coord():
    '''Generates a random coordinate pair inside Finland'''

    country = gpd.read_file('./finland.geojson')
    while True:
        random_x = random.uniform(country.bounds.minx[0], country.bounds.maxx[0])
        random_y = random.uniform(country.bounds.miny[0], country.bounds.maxy[0])
        point = Point(random_x, random_y)

        if country.geometry.contains(point).any():
            return random_x, random_y

def train_model(modelv):
    '''Trains the logistic regression model. There are two models. Model 1 uses the original data and has
    no rescaling. Model 2 uses randomized coordinates for the negative observations and rescales the
    model coefficients for a more realistic model.'''

    df = pd.read_json('./preprocessed.json')
    if modelv == 2:
        # Randomize the coordinates of the negative observations
        df['longitude'] = df.apply(lambda row: random_coord()[0] if row['category'] == 0 else row['longitude'], axis=1)
        df['latitude'] = df.apply(lambda row: random_coord()[1] if row['category'] == 0 else row['latitude'], axis=1)

    features = ['longitude', 'latitude', 'time_from_sunset', 'temperature', 'cloud_cover', 'observation_date']
    X = df[features]
    y = df['category']

    # Use cyclical enconding for the month
    X = X.copy() # This is here to prevent the pandas copy of a slice warning
    X['month'] = X['observation_date'].str.split('-').str[1].astype(int)
    X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
    X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
    X = X.drop(columns=['observation_date', 'month'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Use SMOTE for oversampling the minority class (negative observations)
    oversample = SMOTE(sampling_strategy='minority', random_state=0)
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    # Fit the model
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)

    if modelv == 2:
        # Rescale the model coefficients
        scaling_factors = [0.25, 0.25, 4, 2, 1, 0.5, 0.5]
        scaled_coefficients = model.coef_ * scaling_factors
        model.coef_ = np.array(scaled_coefficients)

    threshold = 0.5
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    coefficients = model.coef_
    print("Coefficients:", coefficients)

    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(model, './logistic_regression_model.pkl')

    return y_test, y_pred, y_prob


def test_model(y_test, y_pred, y_prob):
    '''Tests the performance of the model'''

    conf_matrix = normalize(confusion_matrix(y_test, y_pred), axis=1, norm='l1') 
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("Precision: ", precision)
    print("F1-Score: ", f1)
    print("AUC-ROC: ", auc_roc)
    print(classification_report(y_test, y_pred))

def main():
    y_test1, y_pred1, y_prob1 = train_model(1)
    y_test2, y_pred2, y_prob2 = train_model(2)

    print('Model 1 results:')
    test_model(y_test1, y_pred1, y_prob1)

    print('Model 2 results:')
    test_model(y_test2, y_pred2, y_prob2)

if __name__ == '__main__':
    main()

