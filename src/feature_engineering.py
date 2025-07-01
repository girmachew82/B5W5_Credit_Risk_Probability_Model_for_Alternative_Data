
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Aggregate Features Transformer
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create aggregate features per customer.
    """
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        agg = df.groupby(self.customer_id_col)[self.amount_col].agg([
            ('total_transaction_amount', 'sum'),
            ('average_transaction_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_transaction_amount', 'std')
        ]).reset_index()
        df = df.merge(agg, on=self.customer_id_col, how='left')
        return df

# 2. DateTime Feature Extractor
class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts year, month, day, hour from TransactionStartTime.
    """
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors='coerce')
        df['year'] = df[self.datetime_col].dt.year
        df['month'] = df[self.datetime_col].dt.month
        df['day'] = df[self.datetime_col].dt.day
        df['hour'] = df[self.datetime_col].dt.hour
        return df

# 3. Categorical Encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical variables using OneHot or Label Encoding.
    """
    def __init__(self, categorical_cols, encoding='onehot'):
        self.categorical_cols = categorical_cols
        self.encoding = encoding
        self.encoders = {}

    def fit(self, X, y=None):
        if self.encoding == 'onehot':
            # For scikit-learn >=1.2, use sparse_output; for older, use sparse
            try:
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.encoder.fit(X[self.categorical_cols].astype(str))
        else:
            self.encoders = {}
            for col in self.categorical_cols:
                enc = LabelEncoder()
                enc.fit(X[col].astype(str))
                self.encoders[col] = enc
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.encoding == 'onehot':
            arr = self.encoder.transform(df[self.categorical_cols].astype(str))
            cols = []
            for idx, col in enumerate(self.categorical_cols):
                cats = self.encoder.categories_[idx]
                cols.extend([f"{col}_{cat}" for cat in cats])
            onehot_df = pd.DataFrame(arr, columns=cols, index=df.index)
            df = pd.concat([df.drop(columns=self.categorical_cols), onehot_df], axis=1)
        else:
            for col in self.categorical_cols:
                enc = self.encoders[col]
                df[col] = enc.transform(df[col].astype(str))
        return df

# 4. Missing Value Imputer
class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy_num='mean', strategy_cat='most_frequent'):
        self.strategy_num = strategy_num
        self.strategy_cat = strategy_cat
        self.imputers = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        self.num_cols = df.select_dtypes(include=[np.number]).columns
        self.cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in self.num_cols:
            imp = SimpleImputer(strategy=self.strategy_num)
            imp.fit(df[[col]])
            self.imputers[col] = imp
        for col in self.cat_cols:
            imp = SimpleImputer(strategy=self.strategy_cat)
            imp.fit(df[[col]])
            self.imputers[col] = imp
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col, imp in self.imputers.items():
            # Always flatten the imputed array before assignment
            imputed = imp.transform(df[[col]])
            if imputed.ndim == 2 and imputed.shape[1] == 1:
                df[col] = imputed[:, 0]
            else:
                df[col] = imputed
        return df

# 5. Numeric Scaler
class NumericScaler(BaseEstimator, TransformerMixin):
    """
    Standardize or normalize numeric columns.
    """
    def __init__(self, numeric_cols, scaling='standard'):
        self.numeric_cols = numeric_cols
        self.scaling = scaling
        self.scaler = None

    def fit(self, X, y=None):
        if self.scaling == 'standard':
            self.scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return df

# Pipeline definition
def build_feature_pipeline():
    categorical_cols = [
        'AccountId', 'SubscriptionId', 'CustomerId', 'ProviderId',
        'ProductId', 'ProductCategory', 'ChannelId', 'FraudResult'
    ]
    numeric_cols = ['Amount', 'Value', 'PricingStrategy', 'day', 'hour']
    pipeline = Pipeline([
        ('datetime_features', DateTimeFeatureExtractor(datetime_col='TransactionStartTime')),
        ('aggregate_features', AggregateFeatures(customer_id_col='CustomerId', amount_col='Amount')),
        ('imputer', DataFrameImputer()),
        ('categorical_encoder', CategoricalEncoder(categorical_cols=categorical_cols, encoding='onehot')),
        ('numeric_scaler', NumericScaler(numeric_cols=numeric_cols, scaling='standard'))
    ])
    return pipeline

def process_data(input_csv, output_csv):
    """
    Reads raw data, applies feature engineering pipeline, and saves model-ready data.
    """
    df = pd.read_csv(input_csv)
    pipeline = build_feature_pipeline()
    df_transformed = pipeline.fit_transform(df)
    df_transformed.to_csv(output_csv, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process raw data into model-ready format.")
    parser.add_argument('--input', type=str, required=True, help='Path to raw input CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to save processed CSV')
    args = parser.parse_args()
    process_data(args.input, args.output)
