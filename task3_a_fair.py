import folktables
from folktables import ACSDataSource
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
from itertools import combinations
from sklearn.metrics import accuracy_score
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from aif360.sklearn.metrics import equal_opportunity_difference
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.base import clone


# Set pandas options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust to terminal width
pd.set_option('display.max_colwidth', None)  # Don't truncate column content
# Filter function for the employment prediction task

def employment_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["AGEP"] < 90]
    df = df[df["PWGTP"] >= 1]
    return df

# Define the employment prediction problem
ACSEmployment = folktables.BasicProblem(
    features=[
        "AGEP",  # Age
        "SCHL",  # Educational attainment
        "MAR",   # Marital status
        "RELP",  # Relationship
        "DIS",   # Disability recode
        "ESP",   # Employment status of parents
        "CIT",   # Citizenship status
        "MIG",   # Mobility status (lived here 1 year ago)
        "MIL",   # Military service
        "ANC",   # Ancestry recode
        "NATIVITY",  # Nativity
        "DEAR",  # Hearing difficulty
        "DEYE",  # Vision difficulty
        "DREM",  # Cognitive difficulty
        "SEX",   # Sex
        "RAC1P", # Recoded detailed race code
        "GCL"    # Grandparents living with grandchildren
    ],
    target="ESR",  # Employment status recode
    target_transform=lambda x: x == 1,
    group="DIS",
    preprocess=employment_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

# Load data for Florida state
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["FL"], download=True)

# Convert the data to features, labels, and groups
features, label, group = ACSEmployment.df_to_numpy(acs_data)

data = pd.DataFrame(features, columns=ACSEmployment.features)
data['label'] = label

favorable_classes = [True]
protected_attribute_names = [ACSEmployment.group]
privileged_classes = np.array([[1]])

# Create an AIF360 StandardDataset
data_for_aif = StandardDataset(
    data,
    'label',
    favorable_classes=favorable_classes,
    protected_attribute_names=protected_attribute_names,
    privileged_classes=privileged_classes
)

# Define privileged and unprivileged groups
privileged_groups = [{'DIS': 2}]
unprivileged_groups = [{'DIS': 1}]

df = data
print(df.head())
# List of columns to apply one-hot encoding
#columns_to_encode = ['MAR', 'RELP', 'ESP', 'CIT', 'MIG', 'MIL', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'GCL']
columns_to_encode = ['NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX']

# Perform one-hot encoding for the specified columns
encoded_data = pd.get_dummies(df, columns=columns_to_encode)

binary_dataset = BinaryLabelDataset(
    df=encoded_data,
    label_names=['label'],
    protected_attribute_names=['DIS']
)
reweigher = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
reweigher = reweigher.fit(binary_dataset)
transformed_dataset = reweigher.transform(binary_dataset)

### Testing 
# Add the label column
encoded_data['label'] = transformed_dataset.labels
print(encoded_data.columns)
w_train = transformed_dataset.instance_weights.ravel()

X = data.drop(['label'], axis=1)  # Features
y = data['label']  # Labels

# Initial split into train and test sets
X_train, X_test, y_train, y_test, w_train_split, w_test_split = train_test_split(X, y, w_train, test_size=0.3, random_state=42, shuffle=True)

# Define methods for ranking
methods = {
    'FFS_RF': RandomForestClassifier(random_state=42),
    'RFE_RF': RFE(RandomForestClassifier(random_state=42), n_features_to_select=1),
    'DT_Depth': DecisionTreeClassifier(random_state=42),
    }

# Function to compute rankings and include DIS in the feature rankings
def compute_feature_rankings_with_dis(X, y, demographic_column):
    """
    Computes feature rankings using multiple methods, including the demographic column ('DIS') in the rankings.
    
    Args:
        X (pd.DataFrame): Features dataset (including the demographic column).
        y (pd.Series): Target labels.
        demographic_column (str): Column name for demographics (e.g., 'DIS').

    Returns:
        pd.DataFrame: Final feature rankings for all methods and demographics.
    """
    feature_rankings = {}

    # Get unique demographic groups
    demographic_groups = X[demographic_column].unique()
    demographic_rankings = {}

    for group in demographic_groups:
        print(f"\nProcessing demographic group: {group}")
        # Subset data for the demographic group
        X_subset = X[X[demographic_column] == group]
        y_subset = y[X[demographic_column] == group]

        # Store rankings for each method
        group_rankings = {}
        for method, model in methods.items():
            # Fit the model and calculate feature importance or rankings
            model.fit(X_subset, y_subset)
            if method == 'FFS_RF' or method == 'DT_Depth':
                importances = model.feature_importances_
            elif method == 'RFE_RF':
                importances = model.ranking_
            elif method == 'LogReg_Weights':
                importances = np.abs(model.coef_[0])

            # Sort features by importance and save rankings
            group_rankings[method] = np.argsort(importances)
            print(f"Rankings for {method} (group {group}): {group_rankings[method]}")

        demographic_rankings[group] = group_rankings

    # Combine rankings across methods for each demographic group
    final_demographic_rankings = {}
    for group, rankings in demographic_rankings.items():
        # Compute average ranking for the group across all methods
        average_ranking = np.mean([np.array(rank) for rank in rankings.values()], axis=0)
        final_demographic_rankings[group] = average_ranking
        print(f"Final average ranking for demographic group {group}: {average_ranking}")

    # Combine rankings across demographic groups
    combined_ranking = np.mean(list(final_demographic_rankings.values()), axis=0)
    print(f"\nCombined final ranking across all demographics: {combined_ranking}")

    # Create a DataFrame for the results
    feature_names = X.columns  # Include all features, including 'DIS'
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'Combined_Ranking': combined_ranking
    }).sort_values(by='Combined_Ranking', ascending=True)

    # Add demographic group-specific rankings for analysis
    demographic_results = {
        group: pd.DataFrame({
            'Feature': feature_names,
            'Average_Ranking': final_demographic_rankings[group]
        }).sort_values(by='Average_Ranking', ascending=True)
        for group in demographic_groups
    }

    return ranking_df, demographic_results

# Example usage
final_rankings, demographic_rankings = compute_feature_rankings_with_dis(X, y, demographic_column='DIS')

print("\nCombined Final Rankings Across Demographics:")
print(final_rankings)

# Sort features by Combined_Ranking
sorted_features = final_rankings['Feature'].tolist()

feature_subsets_results = []
#  Fair dt model
# for features in feature_subsets_for_training:

# Loop through the sorted features and create subsets
for i in range(1, len(sorted_features) + 1):  # Start from 1 to include the first feature
    # Create feature subset by selecting top 'i' features
    selected_features = sorted_features[:i]
    DIS_added = False
    X_subset = X_train[selected_features]
    columns_to_encode_in_subset = [col for col in columns_to_encode if col in X_subset.columns] 
    if columns_to_encode_in_subset:
        # Apply one-hot encoding
        X_encoded = pd.get_dummies(X_subset, columns=columns_to_encode_in_subset, drop_first=True)
    else:
        # If no encoding is needed, keep as is
        X_encoded = X_subset.copy()
    # Skip if X_encoded only contains 'DIS' column
    # if set(X_encoded.columns) == {'DIS'}:
    #     print(f"Skipping subset {features} because it only contains 'DIS' column.")
    #     continue
    if 'DIS' not in X_encoded.columns:
        DIS_added = True
        X_encoded['DIS'] = X_train['DIS'] 

    fold_tpr_diff = []
    fold_accuracies = []

    # Perform 5 train-val splits
    train_val_splits = []
    for i in range(5):
        X_train_train, X_train_val, y_train_train, y_train_val, w_train_train_split, w_train_val_split = train_test_split(
            X_encoded, y_train, w_train_split, test_size=0.2, random_state=i, shuffle=True
        )
        train_val_splits.append((X_train_train, X_train_val, y_train_train, y_train_val, w_train_train_split, w_train_val_split))

    # Evaluate each train-val split
    for X_train_train, X_train_val, y_train_train, y_train_val, w_train_train_split, w_train_val_split in train_val_splits:
        # Train the model
        # X_train_train_no_dis = X_train_train.drop(columns=['DIS'])
        # X_train_val_no_dis = X_train_val.drop(columns=['DIS'])

        # Train the model
        val_combined = X_train_val.copy()
        pred_combined = X_train_val.copy()

        if DIS_added:
            X_train_train = X_train_train.drop(columns=['DIS'])
            X_train_val = X_train_val.drop(columns=['DIS'])
        
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)
        model.fit(X_train_train, y_train_train, sample_weight=w_train_train_split)

        # Predict on the validation set
        y_pred = model.predict(X_train_val)

        accuracy = accuracy_score(y_train_val, y_pred)
        fold_accuracies.append(accuracy)
        # Evaluate fairness

        # val_combined = X_train_val.copy()
        val_combined['label'] = y_train_val.values
        # pred_combined = X_train_val.copy()
        pred_combined['label'] = y_pred
        
        # Create datasets for fairness metrics
        test_dataset = BinaryLabelDataset(
            df=val_combined,
            label_names=['label'],
            protected_attribute_names=['DIS']
        )
        pred_dataset = BinaryLabelDataset(
            df=pred_combined,
            label_names=['label'],
            protected_attribute_names=['DIS']
        )
        
        metric = ClassificationMetric(
            test_dataset,
            pred_dataset,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups
        )
        
        # Calculate TPR difference (EOD metric)
        tpr_diff = abs(metric.true_positive_rate(privileged=True) - metric.true_positive_rate(privileged=False))
        fold_tpr_diff.append(tpr_diff)
    
    # Compute average fairness metric for the subset
    avg_eod = np.mean(fold_tpr_diff)
    avg_accuracy = np.mean(fold_accuracies)
    
    print(f"Feature Subset: {selected_features} -> Avg TPR Difference (Fairness): {avg_eod:.4f}")
    feature_subsets_results.append({
        "features": selected_features,
        "avg_eod": avg_eod,
        "avg_accuracy": avg_accuracy
    })

# For Decision Tree, compute the weighted metric and select the best model
best_dt_model = max(feature_subsets_results, key=lambda x: 0.5*x['avg_accuracy'] - 0.5*x['avg_eod'])

best_dt_model_eod = min(feature_subsets_results, key=lambda x: x['avg_eod'])

ranking_eod = sorted(feature_subsets_results, key=lambda x: x['avg_eod'])

feature_subsets_results1 = []
#  Fair dt model
for i in range(1, len(sorted_features) + 1):  # Start from 1 to include the first feature
    # Create feature subset by selecting top 'i' features
    selected_features = sorted_features[:i]
    DIS_added = False
    X_subset = X_train[selected_features]
    columns_to_encode_in_subset = [col for col in columns_to_encode if col in X_subset.columns] 
    if columns_to_encode_in_subset:
        # Apply one-hot encoding
        X_encoded = pd.get_dummies(X_subset, columns=columns_to_encode_in_subset, drop_first=True)
    else:
        # If no encoding is needed, keep as is
        X_encoded = X_subset.copy()
    # Skip if X_encoded only contains 'DIS' column
    # if set(X_encoded.columns) == {'DIS'}:
    #     print(f"Skipping subset {features} because it only contains 'DIS' column.")
    #     continue
    if 'DIS' not in X_encoded.columns:
        DIS_added = True
        X_encoded['DIS'] = X_train['DIS'] 

    fold_tpr_diff = []
    fold_accuracies = []

    # Perform 5 train-val splits
    train_val_splits = []
    for i in range(5):
        X_train_train, X_train_val, y_train_train, y_train_val, w_train_train_split, w_train_val_split = train_test_split(
            X_encoded, y_train, w_train_split, test_size=0.2, random_state=i, shuffle=True
        )
        train_val_splits.append((X_train_train, X_train_val, y_train_train, y_train_val, w_train_train_split, w_train_val_split))

    # Evaluate each train-val split
    for X_train_train, X_train_val, y_train_train, y_train_val, w_train_train_split, w_train_val_split in train_val_splits:
        # Train the model
        # X_train_train_no_dis = X_train_train.drop(columns=['DIS'])
        # X_train_val_no_dis = X_train_val.drop(columns=['DIS'])

        # Train the model
        val_combined = X_train_val.copy()
        pred_combined = X_train_val.copy()

        if DIS_added:
            X_train_train = X_train_train.drop(columns=['DIS'])
            X_train_val = X_train_val.drop(columns=['DIS'])
        
        model = LogisticRegression(C=10, max_iter=100, random_state=42)
        model.fit(X_train_train, y_train_train, sample_weight=w_train_train_split)

        # Predict on the validation set
        y_pred = model.predict(X_train_val)

        accuracy = accuracy_score(y_train_val, y_pred)
        fold_accuracies.append(accuracy)
        # Evaluate fairness

        # val_combined = X_train_val.copy()
        val_combined['label'] = y_train_val.values
        # pred_combined = X_train_val.copy()
        pred_combined['label'] = y_pred
        
        # Create datasets for fairness metrics
        test_dataset = BinaryLabelDataset(
            df=val_combined,
            label_names=['label'],
            protected_attribute_names=['DIS']
        )
        pred_dataset = BinaryLabelDataset(
            df=pred_combined,
            label_names=['label'],
            protected_attribute_names=['DIS']
        )
        
        metric = ClassificationMetric(
            test_dataset,
            pred_dataset,
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups
        )
        
        # Calculate TPR difference (EOD metric)
        tpr_diff = abs(metric.true_positive_rate(privileged=True) - metric.true_positive_rate(privileged=False))
        fold_tpr_diff.append(tpr_diff)
    
    # Compute average fairness metric for the subset
    avg_eod = np.mean(fold_tpr_diff)
    avg_accuracy = np.mean(fold_accuracies)
    
    print(f"Feature Subset: {selected_features} -> Avg TPR Difference (Fairness): {avg_eod:.4f}")
    feature_subsets_results1.append({
        "features": selected_features,
        "avg_eod": avg_eod,
        "avg_accuracy": avg_accuracy
    })


best_lr_model = max(feature_subsets_results1, key=lambda x: 0.5*x['avg_accuracy'] - 0.5*x['avg_eod'])

best_lr_model_eod = min(feature_subsets_results1, key=lambda x: 0.5*x['avg_eod'])

ranking_eod_lr = sorted(feature_subsets_results1, key=lambda x: x['avg_eod'])


import json

model_filename = 'task3a_res_fair//best_dt_model.json'

with open(model_filename, 'w') as json_file:
    json.dump(best_dt_model, json_file, indent=4)  # indent for pretty formatting
print(f"Model saved to {model_filename}")

model_filename = 'task3a_res_fair//best_lr_model.json'

with open(model_filename, 'w') as json_file:
    json.dump(best_lr_model, json_file, indent=4)  # indent for pretty formatting
print(f"Model saved to {model_filename}")


def train_and_evaluate(model_class, params, X_train, y_train, w_train_split, X_test, y_test, protected_attribute, privileged_groups, unprivileged_groups, best_model=None):
    """
    Train the model on the train set and evaluate it on the test set for accuracy and fairness (EOD).
    Optionally, use the feature subset from `best_model` if provided.

    Parameters:
    - model_class: The model class to be instantiated (e.g., DecisionTreeClassifier, LogisticRegression).
    - params: Dictionary of parameters to instantiate the model.
    - X_train: Training features.
    - y_train: Training labels.
    - X_test: Test features.
    - y_test: Test labels.
    - protected_attribute: The name of the protected attribute in the dataset.
    - privileged_groups: Dictionary of privileged groups.
    - unprivileged_groups: Dictionary of unprivileged groups.
    - best_model (optional): Dictionary containing 'features', 'avg_eod', and 'avg_accuracy'. If provided, these features will be used for training and evaluation.

    Returns:
    - A dictionary with model evaluation results: accuracy and EOD.
    """
    # If best_model is provided, use the features from it
    
    selected_features = best_model['features']
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    columns_to_encode_in_subset = [col for col in columns_to_encode if col in X_train.columns] 
    if columns_to_encode_in_subset:
        # Apply one-hot encoding
        X_train_encoded = pd.get_dummies(X_train, columns=columns_to_encode_in_subset, drop_first=True)
    else:
        # If no encoding is needed, keep as is
        X_train_encoded = X_train.copy()

    columns_to_encode_in_subset = [col for col in columns_to_encode if col in X_test.columns] 
    if columns_to_encode_in_subset:
        # Apply one-hot encoding
        X_test_encoded = pd.get_dummies(X_test, columns=columns_to_encode_in_subset, drop_first=True)
    else:
        # If no encoding is needed, keep as is
        X_test_encoded = X_test.copy()

    # Initialize and train the model
    model = model_class(**params)
    model.fit(X_train_encoded, y_train, sample_weight=w_train_split)
    
    # Predict on the test set
    predictions = model.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, predictions)
    
    # Prepare data for fairness evaluation
    test_combined = X_test_encoded.copy()
    test_combined['label'] = y_test.values

    pred_combined = X_test_encoded.copy()
    pred_combined['label'] = predictions

    # Fairness evaluation
    test_dataset = BinaryLabelDataset(
        df=test_combined,
        label_names=['label'],
        protected_attribute_names=[protected_attribute]
    )

    pred_dataset = BinaryLabelDataset(
        df=pred_combined,
        label_names=['label'],
        protected_attribute_names=[protected_attribute]
    )

    metric = ClassificationMetric(
        test_dataset,
        pred_dataset,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups
    )

    eod = abs(metric.true_positive_rate(privileged=True) - metric.true_positive_rate(privileged=False))
    
    return {'accuracy': accuracy, 'eod': eod, 'trained_model': model}

params_lr = {
    'C': 10,  # Example parameter for LogisticRegression
    'max_iter': 100,
    'random_state': 42
}

# Call the train_and_evaluate function
results_lr = train_and_evaluate(
    model_class=LogisticRegression,
    params=params_lr,
    X_train=X_train,
    y_train=y_train,
    w_train_split=w_train_split,
    X_test=X_test,
    y_test=y_test,
    protected_attribute='DIS',  # Example protected attribute
    privileged_groups=privileged_groups,  # Example privileged group
    unprivileged_groups=unprivileged_groups,  # Example unprivileged group
    best_model=best_lr_model  # Passing the best_lr_model
)

# Print the evaluation results
print("Evaluation Results:", results_lr)

trained_model = results_lr['trained_model']
import pickle

# Assuming `model` is your trained model
with open('task3a_res_fair\\trained_lr_model.pkl', 'wb') as file:
    pickle.dump(trained_model, file)

params_dt = {
    'max_depth': 10,  # Example parameter for LogisticRegression
    'min_samples_split': 10,
    'random_state': 42
}

# Call the train_and_evaluate function
results_dt = train_and_evaluate(
    model_class=DecisionTreeClassifier,
    params=params_dt,
    X_train=X_train,
    y_train=y_train,
    w_train_split=w_train_split,
    X_test=X_test,
    y_test=y_test,
    protected_attribute='DIS',  # Example protected attribute
    privileged_groups=privileged_groups,  # Example privileged group
    unprivileged_groups=unprivileged_groups,  # Example unprivileged group
    best_model=best_dt_model  # Passing the best_lr_model
)

# Print the evaluation results
print("Evaluation Results:", results_dt)

trained_model = results_dt['trained_model']
import pickle

# Assuming `model` is your trained model
with open('task3a_res_fair\\trained_dt_model.pkl', 'wb') as file:
    pickle.dump(trained_model, file)


# # Assuming 'data' is your DataFrame and 'DIS' is one of the columns
# correlation_matrix = data.corr()  # Calculate correlation matrix

# # Get the correlation of all columns with 'DIS'
# correlation_with_dis = correlation_matrix['DIS']

# # Split columns into two groups based on the correlation with 'DIS'
# low_corr_cols = correlation_with_dis[correlation_with_dis < 0.3].index.tolist()  # Columns with low correlation (<0.4)
# high_corr_cols = correlation_with_dis[correlation_with_dis >= 0.3].index.tolist()  # Columns with high correlation (>=0.4)

# # Now, `low_corr_cols` contains columns with correlation < 0.4 and `high_corr_cols` contains columns with correlation >= 0.4

# print("Low correlation columns:", low_corr_cols)
# print("High correlation columns:", high_corr_cols)

# # List to store all feature subsets
# feature_subsets_for_training = []

# # Iterate over all subset sizes (1 to len(high_corr_cols))
# for r in range(1, len(high_corr_cols) + 1):
#     # Generate all combinations of `high_corr_cols` of size `r`
#     for combination in itertools.combinations(high_corr_cols, r):
#         # Create a new feature subset by combining the fixed low_corr_cols with the current combination
#         new_ls = list(combination) + low_corr_cols
        
#         # # Ensure 'DIS' is included in the subset
#         # if 'DIS' not in new_ls:
#         #     new_ls.append('DIS')
        
#         # Ensure 'label' is not included in the subset
#         if 'label' in new_ls:
#             new_ls.remove('label')
        
#         # Add the created subset to the list
#         feature_subsets_for_training.append(new_ls)

# feature_subsets_for_training.append(low_corr_cols)
# len(feature_subsets_for_training)