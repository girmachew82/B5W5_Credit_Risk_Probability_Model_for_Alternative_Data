import pandas as pd
from src.feature_engineering import feature_pipeline

def test_aggregate_features():
    data = {
        'CustomerId': [1, 1, 2, 2, 2, 3],
        'Amount': [100, 200, 50, 60, 70, 300]
    }
    df = pd.DataFrame(data)
    result = feature_pipeline.fit_transform(df)
    # Check that aggregate columns exist
    assert 'total_transaction_amount' in result.columns
    assert 'average_transaction_amount' in result.columns
    assert 'transaction_count' in result.columns
    assert 'std_transaction_amount' in result.columns
    # Check values for CustomerId 1
    cust1 = result[result['CustomerId'] == 1].iloc[0]
    assert cust1['total_transaction_amount'] == 300
    assert cust1['average_transaction_amount'] == 150
    assert cust1['transaction_count'] == 2
    assert round(cust1['std_transaction_amount'], 2) == 70.71
