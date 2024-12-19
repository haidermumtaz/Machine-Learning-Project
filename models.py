# models.py
# This file will contain the models for the housing price prediction.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Define a function to perform one-hot encoding, cross-validation, and feature importance analysis
def random_forest_feature_selection(data, target_column, n_splits=10, random_state=42):
    """
    Perform one-hot encoding and feature importance analysis using RandomForestRegressor.

    Parameters:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the target column.
        n_splits (int): Number of splits for cross-validation (default: 10).
        random_state (int): Random state for reproducibility (default: 42).

    Returns:
        pd.DataFrame: Feature importances sorted by importance
        float: Mean R^2 score from cross-validation
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    # Initialize RandomForestRegressor
    model = RandomForestRegressor(random_state=random_state)

    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X_encoded, y, cv=kf, scoring='r2')

    # Get feature importances from a single fit for feature selection purposes
    model.fit(X_encoded, y)
    feature_importances = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return feature_importances, np.mean(scores)

def plot_feature_importance(feature_importances, r2_score, top_n=20):
    """
    Visualize the top N most important features and R^2 score.
    
    Parameters:
        feature_importances (pd.DataFrame): DataFrame with Feature and Importance columns
        r2_score (float): R^2 score to display
        top_n (int): Number of top features to display (default: 20)
    """
    plt.figure(figsize=(12, 8))
    
    
    top_features = feature_importances.head(top_n).iloc[::-1]  
    plt.barh(y=top_features['Feature'], width=top_features['Importance'])
    
    # Add title with R^2 score
    plt.title(f'Top {top_n} Most Important Features for Predicting House Prices\nMean R² Score: {r2_score:.3f}')
    plt.xlabel('Feature Importance Score')
    
    # Add value labels
    for i, v in enumerate(top_features['Importance']):
        plt.text(v, i, f' {v:.3f}')
    
    plt.tight_layout()
    plt.show()

# Define a function to perform K-fold cross-validation and Random Forest model training
def random_forest_regressor(data, target_column, features, n_splits=10, random_state=42):
    """
    Train and evaluate a Random Forest model using K-fold cross-validation.

    Parameters:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the target column.
        features (list): List of features to use for training.
        n_splits (int): Number of splits for cross-validation (default: 10).
        random_state (int): Random state for reproducibility (default: 42).

    Returns:
        pd.DataFrame: DataFrame containing fold-wise scores
        dict: Dictionary containing mean scores
        RandomForestRegressor: Trained model
    """
    # Subset the dataset to include only the specified features
    X = data[features]
    y = data[target_column]

    # Initialize RandomForestRegressor
    model = RandomForestRegressor(random_state=random_state)

    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    mse_scores = -cross_val_score(model, X, y, cv=kf, scoring=mse_scorer)

    # Create DataFrame with fold-wise scores
    scores_df = pd.DataFrame({
        'Fold': range(1, n_splits + 1),
        'R² Score': r2_scores,
        'MSE': mse_scores
    })

    # Calculate mean scores
    mean_scores = {
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mean_mse': np.mean(mse_scores),
        'std_mse': np.std(mse_scores)
    }

    # Fit the model on the full dataset
    model.fit(X, y)

    return scores_df, mean_scores, model

def plot_cv_results(scores_df, mean_scores):
    """
    Visualize cross-validation results with a line plot showing R² scores across folds.
    
    Parameters:
        scores_df (pd.DataFrame): DataFrame containing fold-wise scores
        mean_scores (dict): Dictionary containing mean and std of scores
    """
    plt.figure(figsize=(10, 6))
    
    # Plot R² scores across folds
    plt.plot(scores_df['Fold'], scores_df['R² Score'], 
            'b--o', label='CV R² Scores')
    
    # Add mean R² line
    plt.axhline(y=mean_scores['mean_r2'], color='r', 
                linestyle='-', label=f'Mean R² Score = {mean_scores["mean_r2"]:.4f}')
    
    # Customize the plot
    plt.title('Cross-Validation R² Scores for Multiple Linear Regression Model')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.legend()
    
    # Set y-axis limits with some padding
    max_score = max(scores_df['R² Score'])
    min_score = min(scores_df['R² Score'])
    padding = (max_score - min_score) * 0.1
    plt.ylim(min_score - padding, max_score + padding)
    
    plt.tight_layout()
    plt.show()

    # Print detailed results
    print("\nCross-validation Results:")
    print(f"R² Score: {mean_scores['mean_r2']:.3f} (±{mean_scores['std_r2']:.3f})")
    print(f"MSE: {mean_scores['mean_mse']:.3f} (±{mean_scores['std_mse']:.3f})")

def rf_make_predictions(model, data, features, target_column, log_transformed=False):
    """
    Make predictions using the trained Random Forest model and visualize actual vs predicted values.
    
    Parameters:
        model: Trained RandomForestRegressor model
        data (pd.DataFrame): The input dataset
        features (list): List of features used in training
        target_column (str): Name of the target column
        log_transformed (bool): Whether the target values are log-transformed
    
    Returns:
        pd.DataFrame: DataFrame containing actual and predicted values
    """
    # Make predictions
    X = data[features]
    y_true = data[target_column]
    y_pred = model.predict(X)
    
    # Create DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    # If values were log-transformed, transform them back
    if log_transformed:
        results_df['Actual'] = np.exp(results_df['Actual'])
        results_df['Predicted'] = np.exp(results_df['Predicted'])
    
    # Calculate prediction error
    results_df['Error'] = results_df['Actual'] - results_df['Predicted']
    
    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title('Actual vs Predicted House Prices')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"Mean Absolute Error: ${abs(results_df['Error']).mean():,.2f}")
    print(f"Mean Error: ${results_df['Error'].mean():,.2f}")
    print(f"Error Standard Deviation: ${results_df['Error'].std():,.2f}")
    
    return results_df


