# B5W5 Credit Risk Probability Model for Alternative Data
### Credit Scoring Business Understanding
1. **How the Basel II Accord’s Emphasis on Risk Measurement Influences the Need for an Interpretable and Well-Documented Model:**
    - The Basel II Accord requires financial institutions to generate their own estimates for credit risk parameters like 
        - Probability of Default (PD), 
        - Loss Given Default (LGD), and 
        - Exposure at Default (EAD) for their internal rating-based approaches. 
        - This regulatory focus mandates interpretable and well-documented models to:
            - Demonstrate competency to regulators regarding internal models
            - Support strong model governance frameworks emphasized by regulators like the U.S. 
            - Federal Reserve System (FED), which require conceptual soundness, clear ownership, and regular review.
            - Ensure transparency and explainability.
            - Models, especially complex AI/ML ones, can be "opaque". 
        - However, institutions must explain lending decisions to consumers, auditors, and supervisors, and lack of interpretability can lead to macro-level risks. 
        - Well-documented models provide auditability and compliance with fair lending laws.
2. **Why Creating a Proxy Variable is Necessary When Lacking a Direct "Default" Label, and Potential Business Risks:**
    - When direct loan default data is unavailable, particularly for Micro, Small, and Medium-sized Enterprises (MSMEs) lacking credit history, creating a proxy variable is necessary to develop credit scoring models. For example, delinquency of service charge payments by MSMEs has been used as a proxy for default information to train pre-screening models. This enables financial institutions to assess creditworthiness where traditional data is insufficient. However, using a proxy variable introduces several potential business risks:
        - Inaccurate predictions for true default: The proxy may not perfectly align with actual loan default, leading to unforeseen loan losses.
        - Model risk: Decisions based on an imperfect proxy can contribute to model risk, where incorrect or misused model outputs lead to adverse consequences.
        - Unintended consequences: Flawed creditworthiness understanding can result in approving loans for entities that appear low-risk by the proxy but are high-risk for actual default.
        - Reputational damage: High actual loan defaults or discriminatory outcomes due to a proxy-based assessment can cause significant financial and reputational harm.
3. **Key Trade-offs Between Simple (Logistic Regression) and Complex (Gradient Boosting) Models in a Regulated Financial Context:**
    - Choosing between a simple model like Logistic Regression and a complex one like Gradient Boosting in regulated finance involves critical trade-offs:
        - Logistic Regression (Simple, Interpretable):
            - Pros: It is intuitive, explicable, and generally faster to develop and interpret. Its linear relationship makes logic straightforward to explain to various stakeholders. It's a traditional and widely accepted model for credit scoring.
            - Cons: It may have lower predictive power (AUC scores) compared to advanced algorithms, especially with complex datasets and non-linear relationships.
        - Gradient Boosting (Complex, High-Performance):
            - Pros: Algorithms like XGBoost, LightGBM, and CatBoost generally offer superior predictive power and higher accuracy (AUC scores). They can capture complex, non-linear patterns and handle diverse data effectively.
            - Cons: They are often considered "opaque" or "black box" models, making it challenging to interpret why a specific prediction was made. This lack of interpretability is a significant barrier in regulated financial industries, complicating auditability, regulatory compliance (especially for explainable and fair decisions), and the ability to mitigate issues like algorithmic bias.
#### To do this task
- Development Environment setup
    - Python
    - Git and Github
    - Virtual environment
    - Python packages
        - pandas
        - skisklearn
        - sbseaborn
        - matplotlib
        - mlflow
- EDA
    - `df  = pd.read_csv('../data/raw/data.csv', low_memory=False )`
    - `df.head()`
    - `df.describe(include='all')`
    - `df.shape`
    - `df.info()`
    - `df.isnull().sum()`
    - `df.duplicated().sum()`
    - `df.drop('CurrencyCode', axis=1, inplace=True)`
    - `df.drop('CountryCode', axis=1, inplace=True)`
    - `df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')`
    - `df['year'] = df['TransactionStartTime'].dt.year
    df['month'] = df['TransactionStartTime'].dt.month
    df['day'] = df['TransactionStartTime'].dt.day
    df['hour'] = df['TransactionStartTime'].dt.hour`
    - `numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()`
    - `categorical_cols = df.select_dtypes(include=['object']).columns.tolist()`
    - `for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True, bins=10)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()`
    - `for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    top10 = df[col].value_counts().nlargest(10)
    sns.barplot(x=top10.index.astype(str), y=top10.values)
    plt.title(f"Top 10 Most Frequent Values in {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()`
    - `heatmap_cols = ['Amount', 'Value', 'PricingStrategy', 'day', 'hour']

# Compute the correlation matrix
corr = df[heatmap_cols].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()`
- `numeric_cols = ['Amount', 'Value', 'PricingStrategy', 'day', 'hour']

for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col} (Outlier Detection)')
    plt.xlabel(col)
    plt.show()`
#### Feature Engineering
    # Extract DateTime Features
        `datetime_extractor = DateTimeFeatureExtractor(datetime_col='TransactionStartTime')
        df = datetime_extractor.fit_transform(df)
        df[['year', 'month', 'day', 'hour']].head()`
    # aggegate feature
        `agg_features = AggregateFeatures(customer_id_col='CustomerId', amount_col='Amount')
    df = agg_features.fit_transform(df)
    df[['CustomerId', 'total_transaction_amount', 'average_transaction_amount', 'transaction_count', 'std_transaction_amount']].head()`
#### Model Training
#### Deployment and Testing
