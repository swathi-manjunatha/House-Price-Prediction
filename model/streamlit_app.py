import streamlit as st
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
import os

# Data loading and preprocessing
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "Bengaluru_House_Data.csv")
    df1 = pd.read_csv(data_path)
    #df1 = pd.read_csv("Bengaluru_House_Data.csv")
    df2 = df1.drop(['area_type', 'availability', 'balcony', 'society'], axis='columns')
    df3 = df2.dropna()

    df3['size_new'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

    def is_float(x):
        try:
            float(x)
        except:
            return False
        return True

    df3[~df3['total_sqft'].apply(is_float)]

    def convert_sqft_to_num(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None

    df4 = df3.copy()
    df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)

    df5 = df4.copy()
    df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']

    df5.location = df5.location.apply(lambda x: x.strip())
    location_stats = df5['location'].value_counts(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

    df6 = df5[~((df5.total_sqft / df5.size_new) < 300)]

    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            sd = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft >= (m - sd)) & (subdf.price_per_sqft <= (m + sd))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    df7 = remove_pps_outliers(df6)

    def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('size_new'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('size_new'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')

    df8 = remove_bhk_outliers(df7)
    df9 = df8[df8.bath < df8.size_new + 2]
    df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')

    # One hot encoding
    dummies = pd.get_dummies(df10.location)
    df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
    df12 = df11.drop('location', axis='columns')

    return df12, df5

df12, df5 = load_data()

X = df12.drop('price', axis='columns')
y = df12.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

lr_mod = LinearRegression()
lr_mod.fit(X_train, y_train)

def price_prediction(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    predicted_price = lr_mod.predict([x])[0]
    return max(predicted_price, 0)

# Streamlit UI
st.title("Bengaluru House Price Prediction")

st.write("""
### Enter the details of the house:
""")

sorted_locations = sorted([loc for loc in df5['location'].unique() if loc != 'other'])
default_location_index = sorted_locations.index('Indira Nagar')

location = st.selectbox('Location', sorted_locations, index=default_location_index)
sqft = st.number_input('Total Square Feet', min_value=500)
bath = st.number_input('Number of Bathrooms', min_value=1)
bhk = st.number_input('BHK', min_value=1)

if st.button('Predict Price'):
    price = price_prediction(location, sqft, bath, bhk)
    st.write(f"The estimated price is: ₹{price:.2f} Lakh")

# Uncomment this if you need to show data for debugging
# if st.checkbox('Show Data'):
#     st.write(df1)
