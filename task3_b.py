import folktables
from folktables import ACSDataSource
import numpy as np
import pandas as pd
# Set pandas options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust to terminal width
pd.set_option('display.max_colwidth', None)  # Don't truncate column content
from tqdm import tqdm
import pickle
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from aif360.datasets import StandardDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from aif360.sklearn.metrics import equal_opportunity_difference
import pickle

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



# Load the trained model from the pkl file
with open('task3a_res_fair//trained_lr_model.pkl', 'rb') as file:
    model_lr = pickle.load(file)

# for lr model
features_lr = [
        "RAC1P",
        "GCL",
        "SEX",
        "AGEP",
        "MAR",
        "DEYE",
        "SCHL",
        "ESP",
        "MIL",
        "RELP",
        "DIS",
        "ANC"
    ]
# print(len(features1))
data_lr = data[features_lr]

with open('task3a_res_fair//trained_dt_model.pkl', 'rb') as file:
    model_dt = pickle.load(file)

# for dt model
features_dt = [
        "RAC1P",
        "GCL",
        "SEX",
        "AGEP",
        "MAR",
        "DEYE",
        "SCHL",
        "ESP",
        "MIL",
        "RELP",
        "DIS",
        "ANC",
        "DEAR",
        "CIT",
        "DREM"
    ]
# print(len(features2))
data_dt = data[features_dt]

state_initials = [ "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY" ]
#columns_to_encode = ['MAR', 'RELP', 'ESP', 'CIT', 'MIG', 'MIL', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'GCL']
columns_to_encode = ['NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX']

# Initialize results dictionary
results = {
    "state": [],
    "dt_model_accuracy": [],
    "lr_model_accuracy": [],
    "dt_model_eod": [],
    "lr_model_eod": []
}

# Loop over each state
for state in state_initials:
    print(f"Evaluating models for state: {state}")
    
    # Get data for the current state (excluding Florida)
    acs_data = data_source.get_data(states=[state], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)
    data = pd.DataFrame(features, columns=ACSEmployment.features)
    data['label'] = label

    # Evaluate Normal Model
    X_1 = data[features_lr]
    y_1 = data['label']

    # Apply one-hot encoding if needed
    columns_to_encode_in_subset = [col for col in columns_to_encode if col in X_1.columns]
    if columns_to_encode_in_subset:
        X_1_encoded = pd.get_dummies(X_1, columns=columns_to_encode_in_subset, drop_first=True)
    else:
        X_1_encoded = X_1.copy()

    y_pred_normal = model_lr.predict(X_1_encoded)

    # Evaluate Fair Model
    X_2 = data[features_dt]
    y_2 = data['label']

    columns_to_encode_in_subset = [col for col in columns_to_encode if col in X_2.columns]
    if columns_to_encode_in_subset:
        X_2_encoded = pd.get_dummies(X_2, columns=columns_to_encode_in_subset, drop_first=True)
    else:
        X_2_encoded = X_2.copy()

    y_pred_fair = model_dt.predict(X_2_encoded)
    
    # Calculate accuracy
    accuracy_lr = accuracy_score(y_1, y_pred_normal)
    accuracy_dt = accuracy_score(y_2, y_pred_fair)
    
    # Calculate fairness using Equal Opportunity Difference (EOD)
    val_combined_lr = X_1_encoded.copy()
    val_combined_lr['label'] = y_1.values
    pred_combined_lr = X_1_encoded.copy()
    pred_combined_lr['label'] = y_pred_normal
    
    val_combined_dt = X_2_encoded.copy()
    val_combined_dt['label'] = y_2.values
    pred_combined_dt = X_2_encoded.copy()
    pred_combined_dt['label'] = y_pred_fair
    
    # Create datasets for fairness metrics
    test_dataset_lr = BinaryLabelDataset(df=val_combined_lr, label_names=['label'], protected_attribute_names=['DIS'])
    test_dataset_dt = BinaryLabelDataset(df=val_combined_dt, label_names=['label'], protected_attribute_names=['DIS'])

    pred_dataset_lr = BinaryLabelDataset(df=pred_combined_lr, label_names=['label'], protected_attribute_names=['DIS'])
    pred_dataset_dt = BinaryLabelDataset(df=pred_combined_dt, label_names=['label'], protected_attribute_names=['DIS'])
    
    metric_lr = ClassificationMetric(test_dataset_lr, pred_dataset_lr, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    metric_dt = ClassificationMetric(test_dataset_dt, pred_dataset_dt, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    
    eod_lr = abs(metric_lr.true_positive_rate(privileged=True) - metric_lr.true_positive_rate(privileged=False))
    eod_dt = abs(metric_dt.true_positive_rate(privileged=True) - metric_dt.true_positive_rate(privileged=False))
    
    # Store results
    results["state"].append(state)
    results["lr_model_accuracy"].append(accuracy_lr)
    results["dt_model_accuracy"].append(accuracy_dt)
    results["lr_model_eod"].append(eod_lr)
    results["dt_model_eod"].append(eod_dt)

# Display results
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df)

results_file = "task_3b//results_summary.csv"
results_df.to_csv(results_file, index=False)

print(f"Results summary saved to {results_file}")

results_df = pd.read_csv("task_3b//results_summary.csv")
results_df['FAST_dt'] = 0.5 * results_df['dt_model_accuracy'] + 0.5 * (1 - results_df['dt_model_eod'])
results_df['FAST_lr'] = 0.5 * results_df['lr_model_accuracy'] + 0.5 * (1 - results_df['lr_model_eod'])

import matplotlib.pyplot as plt
import numpy as np

# Data
states = results_df['state']
dt_model_accuracy = results_df['dt_model_accuracy']
lr_model_accuracy = results_df['lr_model_accuracy']
dt_model_eod = results_df['dt_model_eod']
lr_model_eod = results_df['lr_model_eod']
dt_FAST = results_df['FAST_dt']
lr_FAST = results_df['FAST_lr']

# Plot 1: Model Accuracy
plt.figure(figsize=(12, 6))
plt.plot(states, dt_model_accuracy, marker='o', label='DT Model Accuracy', color='blue', linestyle='-')
plt.plot(states, lr_model_accuracy, marker='o', label='LR Model Accuracy', color='orange', linestyle='-')

plt.xlabel('State')
plt.ylabel('Model Accuracy')
plt.title('Model Accuracy by State (Fairness Mitigated)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# Plot 2: Model EOD
plt.figure(figsize=(12, 6))
plt.plot(states, dt_model_eod, marker='o', label='DT Model EOD', color='green', linestyle='-')
plt.plot(states, lr_model_eod, marker='o', label='LR Model EOD', color='red', linestyle='-')

plt.xlabel('State')
plt.ylabel('Model EOD')
plt.title('Model EOD by State (Fairness Mitigated)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# Data
states = results_df['state']
dt_model_accuracy = results_df['dt_model_accuracy']
lr_model_accuracy = results_df['lr_model_accuracy']
dt_model_eod = results_df['dt_model_eod']
lr_model_eod = results_df['lr_model_eod']

# Plot 1: Model FAST
plt.figure(figsize=(12, 6))
plt.plot(states, dt_FAST, marker='o', label='DT Model FAST', color='blue', linestyle='-')
plt.plot(states, lr_FAST, marker='o', label='LR Model FAST', color='orange', linestyle='-')

plt.xlabel('State')
plt.ylabel('Model FAST')
plt.title('Model FAST by State (Fairness Mitigated)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
