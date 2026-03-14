import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    This function implements the core binary encoding logic that converts categorical
    feature with exactly 2 values into 0/1 integers. The mappings are deterministic and must
    be consistent between training and serving.
    """
    
    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna(). unique()).astype(str))
    valset = set(vals)
    
    # Deterministic Binary Mappings
    # yes/No mapping(most common pattern in telecom data)
    if valset == {'Yes', "No"}:
        return s.map({'No': 0, 'Yes': 1}).astype('Int64')
    
    # Gender mapping 
    if valset == {'Male', 'Female'}:
        return s.map({'Male': 1, 'Female': 0}).astype('Int64')
    
    # Generic binary mapping
    if len(vals) == 2:
        # sort values to ensure consistent mapping across runs
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype('Int64')
    
    # Non-binary features
    return s

def build_features(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main features engineering function that transforms raw customer data into
    ML-ready features. The transformations must be exactly replicated in the serving pipeline
    to ensure prediction accuracy.
    """
    df = df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns")
    
    # Identify Feature types
    # Finding categorical columns (object dtype) excluding the target variable
    obj_cols= [c for c in df.select_dtypes(include = ['object']).columns if c != target_col]
    numeric_cols = df.select_dtypes(include = ['int64', 'float64']).columns.tolist()
    
    print(f'Found{len(obj_cols)} categorical and {len(numeric_cols)} numeric columns')
    
    # split the categorical by cardinality
    # Binary features get binary encoding
    # multi-category features get one-hot encoding
    
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    
    print(f' binary features: {len(binary_cols)} | multi-category: {len(multi_cols)}')
    
    if binary_cols:
        print(f'binary: {binary_cols}')
    if multi_cols:
        print(f'multi-category: {multi_cols}')
        
    # Apply Binary Encoding
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f'  {c}: {original_dtype} -> binary (0/1)')
        
    # convert boolean columns
    # XGBoost requires integer inputs, not boolean
    bool_cols = df.select_dtypes(include =['bool']).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f'converted {len(bool_cols)} boolean columns to int: {bool_cols}')
        
    # one-hot encoding for multi-category features
    if multi_cols:
        print(f'applying one-hot encoding to {len(multi_cols)} multi-category columns')
        original_shape = df.shape
        
        # apply one-hot encoding with drop_first = True
        df = pd.get_dummies(df, columns = multi_cols, drop_first= True)
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print('created {new_features} new features from {len(multi_cols)} categorical columns')
        
    # data type cleanup
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)
            
    print(f'Feature engineering complete: {df.shape[1]} final features')
    return df                       
    