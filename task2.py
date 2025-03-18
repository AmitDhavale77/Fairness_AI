import folktables
from folktables import ACSDataSource
import numpy as np
import pandas as pd
# Set pandas options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust to terminal width
pd.set_option('display.max_colwidth', None)  # Don't truncate column content
from tqdm import tqdm
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
import pandas as pd
from aif360.datasets import StandardDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from aif360.sklearn.metrics import equal_opportunity_difference

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
# Define fairness-related groups
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

print(data)
df = data

print(df.head())
# List of columns to apply one-hot encoding
columns_to_encode = ['MAR', 'RELP', 'ESP', 'CIT', 'MIG', 'MIL', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'GCL']

# Perform one-hot encoding for the specified columns
encoded_data = pd.get_dummies(df, columns=columns_to_encode)

# Show the resulting dataframe
print(encoded_data)
encoded_data['label'] = label

# Convert to BinaryLabelDataset
binary_dataset = BinaryLabelDataset(
    df=encoded_data,
    label_names=['label'],
    protected_attribute_names=['DIS']
)

reweigher = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
reweigher = reweigher.fit(binary_dataset)
transformed_dataset = reweigher.transform(binary_dataset)

# metric_transf_train = BinaryLabelDatasetMetric(transformed_dataset, 
#                                          unprivileged_groups=unprivileged_groups,
#                                          privileged_groups=privileged_groups)

# Add the label column
encoded_data['label'] = transformed_dataset.labels

print(encoded_data.columns)

X = encoded_data.drop(['label'], axis=1)
y = encoded_data['label'] 

w_train = transformed_dataset.instance_weights.ravel()

# Split the data and weights together
X_train, X_test, y_train, y_test, w_train_split, w_test_split = train_test_split(X, y, w_train, test_size=0.3, random_state=42, shuffle=True)

train_val_splits = []
for i in range(5):  # Perform 5 splits
    X_train_train, X_train_val, y_train_train, y_train_val, w_train_train_split, w_test_test_split = train_test_split(
        X_train, y_train, w_train_split, test_size=0.2, random_state=i, shuffle=True
    )
    train_val_splits.append((X_train_train, X_train_val, y_train_train, y_train_val, w_train_train_split, w_test_test_split))

# Hyperparameter grids
dt_params = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
lr_params = {'C': [0.1, 1, 10], 'max_iter': [50, 100, 150]}

# Store results
results_dt = []
results_lr = []

# Decision Tree Hyperparameter Tuning
for max_depth in dt_params['max_depth']:
    for min_samples_split in dt_params['min_samples_split']:
        accuracies = []
        fold_tpr_diff = []
        for X_train_train, X_train_val, y_train_train, y_train_val, w_train_train_split, w_test_test_split in train_val_splits:
            # Train the model
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            model.fit(X_train_train, y_train_train, sample_weight=w_train_train_split)
            
            # Predict without altering X_train_val
            y_pred = model.predict(X_train_val)

            # Calculate accuracy
            accuracies.append(accuracy_score(y_train_val, y_pred))

            # Prepare data for fairness evaluation
            val_combined = X_train_val.copy()  # Create a copy for fairness metrics
            val_combined['label'] = y_train_val.values  # Add true labels
            
            pred_combined = X_train_val.copy()  # Create a copy for fairness metrics
            pred_combined['label'] = y_pred  # Add predicted labels

            # Create BinaryLabelDataset for true labels
            test_dataset = BinaryLabelDataset(
                df=val_combined,
                label_names=['label'],
                protected_attribute_names=['DIS']
            )

            # Create BinaryLabelDataset for predicted labels
            pred_dataset = BinaryLabelDataset(
                df=pred_combined,
                label_names=['label'],
                protected_attribute_names=['DIS']
            )

            # Evaluate fairness using ClassificationMetric
            metric = ClassificationMetric(
                test_dataset,
                pred_dataset,
                privileged_groups=privileged_groups,
                unprivileged_groups=unprivileged_groups
            )

            # Calculate TPR difference
            tpr_diff = abs(metric.true_positive_rate(privileged=True) - metric.true_positive_rate(privileged=False))
            fold_tpr_diff.append(tpr_diff)

        # Calculate average accuracy and EOD for the hyperparameter combination
        avg_accuracy = np.mean(accuracies)
        avg_eod = np.mean(fold_tpr_diff)
        print(f"dt - Accuracy: {avg_accuracy}, EOD: {avg_eod}")

        # Save results
        results_dt.append({
            'model': 'DecisionTree',
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'accuracy': avg_accuracy,
            'eod': avg_eod
        })


# Logistic Regression Hyperparameter Tuning
for C in lr_params['C']:
    for max_iter in lr_params['max_iter']:
        accuracies = []
        fold_tpr_diff = []
        for X_train_train, X_train_val, y_train_train, y_train_val, w_train_train_split, w_test_test_split in train_val_splits:
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
            model.fit(X_train_train, y_train_train, sample_weight=w_train_train_split)
            y_pred = model.predict(X_train_val)

            # Calculate accuracy
            accuracies.append(accuracy_score(y_train_val, y_pred))

            # Prepare data for fairness evaluation
            val_combined = X_train_val.copy()  # Create a copy for fairness metrics
            val_combined['label'] = y_train_val.values  # Add true labels
            
            pred_combined = X_train_val.copy()  # Create a copy for fairness metrics
            pred_combined['label'] = y_pred  # Add predicted labels

            # Create BinaryLabelDataset for true labels
            test_dataset = BinaryLabelDataset(
                df=val_combined,
                label_names=['label'],
                protected_attribute_names=['DIS']
            )

            # Create BinaryLabelDataset for predicted labels
            pred_dataset = BinaryLabelDataset(
                df=pred_combined,
                label_names=['label'],
                protected_attribute_names=['DIS']
            )

            # Evaluate fairness using ClassificationMetric
            metric = ClassificationMetric(
                test_dataset,
                pred_dataset,
                privileged_groups=privileged_groups,
                unprivileged_groups=unprivileged_groups
            )

            # Calculate TPR difference
            tpr_diff = abs(metric.true_positive_rate(privileged=True) - metric.true_positive_rate(privileged=False))
            fold_tpr_diff.append(tpr_diff)

        # Calculate average accuracy and EOD for the hyperparameter combination
        avg_accuracy = np.mean(accuracies)
        avg_eod = np.mean(fold_tpr_diff)
        print(f"lr - Accuracy: {avg_accuracy}, EOD: {avg_eod}")

        results_lr.append({'model': 'LogisticRegression', 'C': C, 'accuracy': avg_accuracy, 'eod': avg_eod, 'max_iter': max_iter})

# Convert results into DataFrames for easier processing
df_dt = pd.DataFrame(results_dt)
df_lr = pd.DataFrame(results_lr)

import matplotlib.pyplot as plt
import seaborn as sns

# Create a combined hyperparameter column for Decision Tree
df_dt['hyperparams'] = df_dt.apply(lambda row: f"Depth={row['max_depth']}, Split={row['min_samples_split']}", axis=1)

# Create a combined hyperparameter column for Logistic Regression
df_lr['hyperparams'] = df_lr.apply(lambda row: f"C={row['C']}, Iter={row['max_iter']}", axis=1)

# Set style
sns.set(style="whitegrid")

# Decision Tree: Accuracy and Fairness Plot
plt.figure(figsize=(14, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_dt, x='hyperparams', y='accuracy', marker='o', color='blue', label="Accuracy")
sns.lineplot(data=df_dt, x='hyperparams', y='accuracy', color='blue')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Max Depth & Min Samples Split')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Accuracy vs Hyperparameters (After Reweighting)')

# Fairness (EOD) plot
plt.subplot(1, 2, 2)
sns.scatterplot(data=df_dt, x='hyperparams', y='eod', marker='o', color='red', label="EOD")
sns.lineplot(data=df_dt, x='hyperparams', y='eod', color='red')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Max Depth & Min Samples Split')
plt.ylabel('Equal Opportunity Difference (EOD)')
plt.title('Decision Tree: Fairness vs Hyperparameters (After Reweighting)')

plt.tight_layout()
plt.show()

# Logistic Regression: Accuracy and Fairness Plot
plt.figure(figsize=(14, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_lr, x='hyperparams', y='accuracy', marker='o', color='blue', label="Accuracy")
sns.lineplot(data=df_lr, x='hyperparams', y='accuracy', color='blue')
plt.xticks(rotation=45, ha='right')
plt.xlabel('C & Max Iterations')
plt.ylabel('Accuracy')
plt.title('Logistic Regression: Accuracy vs Hyperparameters (After Reweighting)')

# Fairness (EOD) plot
plt.subplot(1, 2, 2)
sns.scatterplot(data=df_lr, x='hyperparams', y='eod', marker='o', color='red', label="EOD")
sns.lineplot(data=df_lr, x='hyperparams', y='eod', color='red')
plt.xticks(rotation=45, ha='right')
plt.xlabel('C & Max Iterations')
plt.ylabel('Equal Opportunity Difference (EOD)')
plt.title('Logistic Regression: Fairness vs Hyperparameters (After Reweighting)')

plt.tight_layout()
plt.show()

# Function to select the best models based on accuracy and EOD
def select_top_models(results_df, model_name):
    # Get the model with the highest accuracy
    best_accuracy_model = results_df.sort_values(by='accuracy', ascending=False).iloc[0]
    
    # Get the model with the lowest EOD
    best_eod_model = results_df.sort_values(by='eod', ascending=True).iloc[0]
    
    print(f"Best {model_name} model based on accuracy:\n{best_accuracy_model}\n")
    print(f"Best {model_name} model based on EOD:\n{best_eod_model}\n")
    
    return best_accuracy_model, best_eod_model

# Select the top models for Decision Tree
best_dt_accuracy, best_dt_eod = select_top_models(df_dt, 'DecisionTree')

# Select the top models for Logistic Regression
best_lr_accuracy, best_lr_eod = select_top_models(df_lr, 'LogisticRegression')

# Combine the results into a single summary
best_models_summary = pd.DataFrame([
    best_dt_accuracy,
    best_dt_eod,
    best_lr_accuracy,
    best_lr_eod
])

# Save the summary to a CSV file
summary_file = "results_t2//best_models_summary.csv"
best_models_summary.to_csv(summary_file, index=False)

print(f"Best models summary saved to {summary_file}")

def train_and_evaluate(model_class, params, X_train, y_train, w_train_split, X_test, y_test, protected_attribute, privileged_groups, unprivileged_groups):
    """
    Train the model on the train set and evaluate it on the test set for accuracy and fairness (EOD).

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

    Returns:
    - A dictionary with model evaluation results: accuracy and EOD.
    """
    # Initialize and train the model
    model = model_class(**params)
    model.fit(X_train, y_train, sample_weight=w_train_split)
    
    # Predict on the test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Prepare data for fairness evaluation
    test_combined = X_test.copy()
    test_combined['label'] = y_test.values

    pred_combined = X_test.copy()
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
    
    return {'accuracy': accuracy, 'eod': eod}

# Select the models with the best accuracy
best_dt_model = best_dt_accuracy
best_lr_model = best_lr_accuracy

# Select the models with the best accuracy
best_dt_model = best_dt_accuracy
best_lr_model = best_lr_accuracy

# Extract the best parameters for Decision Tree and Logistic Regression
dt_params = {
    'max_depth': best_dt_model['max_depth'],
    'min_samples_split': best_dt_model['min_samples_split'],
    'random_state': 42
}

lr_params = {
    'C': best_lr_model['C'],
    'max_iter': best_lr_model['max_iter'],
    'random_state': 42
}


# Evaluate the Decision Tree with best fairness
dt_fairness_results = train_and_evaluate(
    model_class=DecisionTreeClassifier,
    params=dt_params,
    X_train=X_train,
    y_train=y_train,
    w_train_split=w_train_split,
    X_test=X_test,
    y_test=y_test,
    protected_attribute='DIS',
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)

print(f"Decision Tree (Accuracy) - Test Accuracy: {dt_fairness_results['accuracy']}, Test EOD: {dt_fairness_results['eod']}")

# Evaluate the Logistic Regression with best accuracy
lr_fairness_results = train_and_evaluate(
    model_class=LogisticRegression,     # Pass the model class
    params=lr_params,                   # Pass the dictionary of parameters
    X_train=X_train,
    y_train=y_train,
    w_train_split=w_train_split,
    X_test=X_test,
    y_test=y_test,
    protected_attribute='DIS',
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)

print(f"Logistic Regression (Best Accuracy) - Test Accuracy: {lr_fairness_results['accuracy']}, Test EOD: {lr_fairness_results['eod']}")

# Models with the best fairness
best_dt_fairness_params = {
    'max_depth': best_dt_eod['max_depth'],
    'min_samples_split': best_dt_eod['min_samples_split'],
    'random_state': 42
}

best_lr_fairness_params = {
    'C': best_lr_eod['C'],
    'max_iter': best_lr_eod['max_iter'],
    'random_state': 42
}

# Evaluate the Decision Tree with best fairness
dt_fairness_results = train_and_evaluate(
    model_class=DecisionTreeClassifier,  # Pass the class, not an instance
    params=best_dt_fairness_params,
    X_train=X_train,
    y_train=y_train,
    w_train_split=w_train_split,
    X_test=X_test,
    y_test=y_test,
    protected_attribute='DIS',
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)

print(f"Decision Tree (Fairness) - Test Accuracy: {dt_fairness_results['accuracy']}, Test EOD: {dt_fairness_results['eod']}")

# Evaluate the Logistic Regression with best fairness
lr_fairness_results = train_and_evaluate(
    model_class=LogisticRegression,  # Pass the class, not an instance
    params=best_lr_fairness_params,
    X_train=X_train,
    y_train=y_train,
    w_train_split=w_train_split,
    X_test=X_test,
    y_test=y_test,
    protected_attribute='DIS',
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)

print(f"Logistic Regression (Fairness) - Test Accuracy: {lr_fairness_results['accuracy']}, Test EOD: {lr_fairness_results['eod']}")

