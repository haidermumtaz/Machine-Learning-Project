# models.py
# This file will contain the models for the housing price prediction.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


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
        data (pd.DataFrame): The input dataset
        target_column (str): Name of the target column
        features (list): List of features to use
        n_splits (int): Number of splits for cross-validation
        random_state (int): Random state for reproducibility
    
    Returns:
        dict: Dictionary containing model, scores, and feature importance
    """
    # Prepare the data
    X = data[features]
    y = data[target_column]
    
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=random_state
    )
    
    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        r2_scores.append(r2_score(y_val, y_pred))
        mse_scores.append(mean_squared_error(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
    
    # Train final model on full dataset
    final_model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=random_state
    )
    
    final_model.fit(X, y)
    
    # Calculate mean scores
    mean_scores = {
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mean_mse': np.mean(mse_scores),
        'std_mse': np.std(mse_scores),
        'mean_mae': np.mean(mae_scores),
        'std_mae': np.std(mae_scores)
    }
    
    # Store fold-wise scores
    scores = {
        'r2_scores': r2_scores,
        'mse_scores': mse_scores,
        'mae_scores': mae_scores
    }
    
    return {
        'model': final_model,
        'scores': mean_scores,
        'fold_scores': scores,
        'feature_importance': pd.DataFrame({
            'Feature': features,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    }

def plot_cv_results(scores, mean_scores):
    """
    Visualize cross-validation results with a line plot showing R² scores across folds.
    
    Parameters:
        scores (dict): Dictionary containing fold-wise scores
        mean_scores (dict): Dictionary containing mean and std of scores
    """
    plt.figure(figsize=(10, 6))
    
    # Create fold numbers for x-axis
    folds = range(1, len(scores['r2_scores']) + 1)
    
    # Plot R² scores across folds
    plt.plot(folds, scores['r2_scores'], 
            'b--o', label='CV R² Scores')
    
    # Add mean R² line
    plt.axhline(y=mean_scores['mean_r2'], color='r', 
                linestyle='-', label=f'Mean R² Score = {mean_scores["mean_r2"]:.4f}')
    
    # Customize the plot
    plt.title('Cross-Validation R² Scores for Random Forest')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.legend()
    
    # Set y-axis limits with some padding
    max_score = max(scores['r2_scores'])
    min_score = min(scores['r2_scores'])
    padding = (max_score - min_score) * 0.1
    plt.ylim(min_score - padding, max_score + padding)
    
    plt.tight_layout()
    plt.show()

    # Print detailed results
    print("\nCross-validation Results:")
    print(f"R² Score: {mean_scores['mean_r2']:.3f} (±{mean_scores['std_r2']:.3f})")
    print(f"MSE: {mean_scores['mean_mse']:.3f} (±{mean_scores['std_mse']:.3f})")
    print(f"MAE: {mean_scores['mean_mae']:.3f} (±{mean_scores['std_mae']:.3f})")

def rf_make_predictions(model, data, features, target_column, log_transformed=False):
    """
    Make predictions using the trained Random Forest model and visualize actual vs predicted values.
    
    Parameters:
        model: Trained Random Forest model (from random_forest_regressor results['model'])
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
    
    plt.title('Actual vs Predicted House Prices (Random Forest)')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    raw_mean_error = results_df['Error'].mean()
    print("\nPrediction Summary:")
    print(f"Mean Absolute Error: ${abs(results_df['Error']).mean():,.2f}")
    print(f"Mean Error: ${raw_mean_error:,.2f}")
    print(f"Error Standard Deviation: ${results_df['Error'].std():,.2f}")
    
    return results_df

def xgb_train_model(data, target_column, features, n_splits=10, random_state=42):
    """
    Train and evaluate an XGBoost model using K-fold cross-validation.
    
    Parameters:
        data (pd.DataFrame): The input dataset
        target_column (str): Name of the target column
        features (list): List of features to use
        n_splits (int): Number of splits for cross-validation
        random_state (int): Random state for reproducibility
    
    Returns:
        dict: Dictionary containing scores and trained model
    """
    # Prepare the data
    X = data[features]
    y = data[target_column]
    
    # Initialize model with good default parameters for house prices
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    
    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model - simplified without early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        r2_scores.append(r2_score(y_val, y_pred))
        mse_scores.append(mean_squared_error(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
    
    # Train final model on full dataset
    final_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    
    final_model.fit(X, y)
    
    # Calculate mean scores
    mean_scores = {
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mean_mse': np.mean(mse_scores),
        'std_mse': np.std(mse_scores),
        'mean_mae': np.mean(mae_scores),
        'std_mae': np.std(mae_scores)
    }
    
    # Store fold-wise scores
    scores = {
        'r2_scores': r2_scores,
        'mse_scores': mse_scores,
        'mae_scores': mae_scores
    }
    
    return {
        'model': final_model,
        'scores': mean_scores,
        'fold_scores': scores,
        'feature_importance': pd.DataFrame({
            'Feature': features,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    }

def plot_xgb_results(results):
    """
    Plot feature importance and print model performance metrics.
    
    Parameters:
        results (dict): Dictionary containing model results
    """
    scores = results['scores']
    
    # Print performance metrics
    print("\nXGBoost Model Performance:")
    print(f"R² Score: {scores['mean_r2']:.3f} (±{scores['std_r2']:.3f})")
    print(f"MSE: {scores['mean_mse']:.3f} (±{scores['std_mse']:.3f})")
    print(f"MAE: {scores['mean_mae']:.3f} (±{scores['std_mae']:.3f})")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    importance_df = results['feature_importance'].head(20).iloc[::-1]  # Added iloc[::-1] to reverse order
    
    plt.barh(y=importance_df['Feature'], width=importance_df['Importance'])
    plt.title('Top 20 Most Important Features (XGBoost)')
    plt.xlabel('Feature Importance Score')
    
    plt.tight_layout()
    plt.show()

def xgb_make_predictions(model, data, features, target_column, log_transformed=False):
    """
    Make predictions using the trained XGBoost model and visualize actual vs predicted values.
    
    Parameters:
        model: Trained XGBoost model (from xgb_train_model results['model'])
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
    
    plt.title('Actual vs Predicted House Prices (XGBoost)')
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

def plot_xgb_cv_results(scores, mean_scores):
    """
    Visualize cross-validation results for XGBoost with a line plot showing R² scores across folds.
    
    Parameters:
        scores (dict): Dictionary containing fold-wise scores
        mean_scores (dict): Dictionary containing mean and std of scores
    """
    plt.figure(figsize=(10, 6))
    
    # Create fold numbers for x-axis
    folds = range(1, len(scores['r2_scores']) + 1)
    
    # Plot R² scores across folds
    plt.plot(folds, scores['r2_scores'], 
            'b--o', label='CV R² Scores')
    
    # Add mean R² line
    plt.axhline(y=mean_scores['mean_r2'], color='r', 
                linestyle='-', label=f'Mean R² Score = {mean_scores["mean_r2"]:.4f}')
    
    # Customize the plot
    plt.title('Cross-Validation R² Scores for XGBoost Model')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.legend()
    
    # Set y-axis limits with some padding
    max_score = max(scores['r2_scores'])
    min_score = min(scores['r2_scores'])
    padding = (max_score - min_score) * 0.1
    plt.ylim(min_score - padding, max_score + padding)
    
    plt.tight_layout()
    plt.show()

    # Print detailed results
    print("\nCross-validation Results:")
    print(f"R² Score: {mean_scores['mean_r2']:.3f} (±{mean_scores['std_r2']:.3f})")
    print(f"MSE: {mean_scores['mean_mse']:.3f} (±{mean_scores['std_mse']:.3f})")
    print(f"MAE: {mean_scores['mean_mae']:.3f} (±{mean_scores['std_mae']:.3f})")

def lgb_train_model(data, target_column, features, n_splits=10, random_state=42):
    """
    Train and evaluate a LightGBM model using K-fold cross-validation.
    
    Parameters:
        data (pd.DataFrame): The input dataset
        target_column (str): Name of the target column
        features (list): List of features to use
        n_splits (int): Number of splits for cross-validation
        random_state (int): Random state for reproducibility
    
    Returns:
        dict: Dictionary containing model, scores, and feature importance
    """
    # Prepare the data
    X = data[features]
    y = data[target_column]
    
    # Initialize model with good default parameters for house prices
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1
    )
    
    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model - simplified fit call
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        r2_scores.append(r2_score(y_val, y_pred))
        mse_scores.append(mean_squared_error(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
    
    # Train final model on full dataset
    final_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1
    )
    
    final_model.fit(X, y)
    
    # Calculate mean scores
    mean_scores = {
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mean_mse': np.mean(mse_scores),
        'std_mse': np.std(mse_scores),
        'mean_mae': np.mean(mae_scores),
        'std_mae': np.std(mae_scores)
    }
    
    # Store fold-wise scores
    scores = {
        'r2_scores': r2_scores,
        'mse_scores': mse_scores,
        'mae_scores': mae_scores
    }
    
    return {
        'model': final_model,
        'scores': mean_scores,
        'fold_scores': scores,
        'feature_importance': pd.DataFrame({
            'Feature': features,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    }

def plot_lgb_results(results):
    """
    Plot feature importance and print model performance metrics for LightGBM.
    
    Parameters:
        results (dict): Dictionary containing model results
    """
    scores = results['scores']
    
    # Print performance metrics
    print("\nLightGBM Model Performance:")
    print(f"R² Score: {scores['mean_r2']:.3f} (±{scores['std_r2']:.3f})")
    print(f"MSE: {scores['mean_mse']:.3f} (±{scores['std_mse']:.3f})")
    print(f"MAE: {scores['mean_mae']:.3f} (±{scores['std_mae']:.3f})")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    importance_df = results['feature_importance'].head(20).iloc[::-1]  # Reverse order
    
    plt.barh(y=importance_df['Feature'], width=importance_df['Importance'])
    plt.title('Top 20 Most Important Features (LightGBM)')
    plt.xlabel('Feature Importance Score')
    
    plt.tight_layout()
    plt.show()

def plot_lgb_cv_results(scores, mean_scores):
    """
    Visualize cross-validation results for LightGBM with a line plot showing R² scores across folds.
    
    Parameters:
        scores (dict): Dictionary containing fold-wise scores
        mean_scores (dict): Dictionary containing mean and std of scores
    """
    plt.figure(figsize=(10, 6))
    
    # Create fold numbers for x-axis
    folds = range(1, len(scores['r2_scores']) + 1)
    
    # Plot R² scores across folds
    plt.plot(folds, scores['r2_scores'], 
            'b--o', label='CV R² Scores')
    
    # Add mean R² line
    plt.axhline(y=mean_scores['mean_r2'], color='r', 
                linestyle='-', label=f'Mean R² Score = {mean_scores["mean_r2"]:.4f}')
    
    # Customize the plot
    plt.title('Cross-Validation R² Scores for LightGBM Model')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.legend()
    
    # Set y-axis limits with some padding
    max_score = max(scores['r2_scores'])
    min_score = min(scores['r2_scores'])
    padding = (max_score - min_score) * 0.1
    plt.ylim(min_score - padding, max_score + padding)
    
    plt.tight_layout()
    plt.show()

def lgb_make_predictions(model, data, features, target_column, log_transformed=False):
    """
    Make predictions using the trained LightGBM model and visualize actual vs predicted values.
    
    Parameters:
        model: Trained LightGBM model (from lgb_train_model results['model'])
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
    
    plt.title('Actual vs Predicted House Prices (LightGBM)')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics with raw mean error value
    raw_mean_error = results_df['Error'].mean()
    print("\nPrediction Summary:")
    print(f"Mean Absolute Error: ${abs(results_df['Error']).mean():,.2f}")
    print(f"Mean Error: ${raw_mean_error:,.2f}")  # Previous format
    print(f"Error Standard Deviation: ${results_df['Error'].std():,.2f}")
    
    return results_df

def catboost_train_model(data, target_column, features, n_splits=10, random_state=42):
    """
    Train and evaluate a CatBoost model using K-fold cross-validation.
    
    Parameters:
        data (pd.DataFrame): The input dataset
        target_column (str): Name of the target column
        features (list): List of features to use
        n_splits (int): Number of splits for cross-validation
        random_state (int): Random state for reproducibility
    
    Returns:
        dict: Dictionary containing model, scores, and feature importance
    """
    # Prepare the data
    X = data[features]
    y = data[target_column]
    
    # Initialize model with good default parameters for house prices
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        subsample=0.8,
        random_seed=random_state,
        verbose=False  # Suppress training output
    )
    
    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_train, y_train, eval_set=(X_val, y_val), silent=True)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        r2_scores.append(r2_score(y_val, y_pred))
        mse_scores.append(mean_squared_error(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
    
    # Train final model on full dataset
    final_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        subsample=0.8,
        random_seed=random_state,
        verbose=False
    )
    
    final_model.fit(X, y, silent=True)
    
    # Calculate mean scores
    mean_scores = {
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mean_mse': np.mean(mse_scores),
        'std_mse': np.std(mse_scores),
        'mean_mae': np.mean(mae_scores),
        'std_mae': np.std(mae_scores)
    }
    
    # Store fold-wise scores
    scores = {
        'r2_scores': r2_scores,
        'mse_scores': mse_scores,
        'mae_scores': mae_scores
    }
    
    return {
        'model': final_model,
        'scores': mean_scores,
        'fold_scores': scores,
        'feature_importance': pd.DataFrame({
            'Feature': features,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    }

def plot_catboost_cv_results(scores, mean_scores):
    """
    Visualize cross-validation results for CatBoost with a line plot showing R² scores across folds.
    
    Parameters:
        scores (dict): Dictionary containing fold-wise scores
        mean_scores (dict): Dictionary containing mean and std of scores
    """
    plt.figure(figsize=(10, 6))
    
    # Create fold numbers for x-axis
    folds = range(1, len(scores['r2_scores']) + 1)
    
    # Plot R² scores across folds
    plt.plot(folds, scores['r2_scores'], 
            'b--o', label='CV R² Scores')
    
    # Add mean R² line
    plt.axhline(y=mean_scores['mean_r2'], color='r', 
                linestyle='-', label=f'Mean R² Score = {mean_scores["mean_r2"]:.4f}')
    
    # Customize the plot
    plt.title('Cross-Validation R² Scores for CatBoost Model')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.legend()
    
    # Set y-axis limits with some padding
    max_score = max(scores['r2_scores'])
    min_score = min(scores['r2_scores'])
    padding = (max_score - min_score) * 0.1
    plt.ylim(min_score - padding, max_score + padding)
    
    plt.tight_layout()
    plt.show()

    # Print detailed results
    print("\nCross-validation Results:")
    print(f"R² Score: {mean_scores['mean_r2']:.3f} (±{mean_scores['std_r2']:.3f})")
    print(f"MSE: {mean_scores['mean_mse']:.3f} (±{mean_scores['std_mse']:.3f})")
    print(f"MAE: {mean_scores['mean_mae']:.3f} (±{mean_scores['std_mae']:.3f})")

def catboost_make_predictions(model, data, features, target_column, log_transformed=False):
    """
    Make predictions using the trained CatBoost model and visualize actual vs predicted values.
    
    Parameters:
        model: Trained CatBoost model (from catboost_train_model results['model'])
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
    
    plt.title('Actual vs Predicted House Prices (CatBoost)')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    raw_mean_error = results_df['Error'].mean()
    print("\nPrediction Summary:")
    print(f"Mean Absolute Error: ${abs(results_df['Error']).mean():,.2f}")
    print(f"Mean Error: ${raw_mean_error:,.2f}")
    print(f"Error Standard Deviation: ${results_df['Error'].std():,.2f}")
    
    return results_df

def extra_trees_train_model(data, target_column, features, n_splits=10, random_state=42):
    """
    Train and evaluate an Extra Trees model using K-fold cross-validation.
    
    Parameters:
        data (pd.DataFrame): The input dataset
        target_column (str): Name of the target column
        features (list): List of features to use
        n_splits (int): Number of splits for cross-validation
        random_state (int): Random state for reproducibility
    
    Returns:
        dict: Dictionary containing model, scores, and feature importance
    """
    # Prepare the data
    X = data[features]
    y = data[target_column]
    
    # Initialize model
    model = ExtraTreesRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=random_state
    )
    
    # Perform K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        r2_scores.append(r2_score(y_val, y_pred))
        mse_scores.append(mean_squared_error(y_val, y_pred))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
    
    # Train final model on full dataset
    final_model = ExtraTreesRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=random_state
    )
    
    final_model.fit(X, y)
    
    # Calculate mean scores
    mean_scores = {
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'mean_mse': np.mean(mse_scores),
        'std_mse': np.std(mse_scores),
        'mean_mae': np.mean(mae_scores),
        'std_mae': np.std(mae_scores)
    }
    
    # Store fold-wise scores
    scores = {
        'r2_scores': r2_scores,
        'mse_scores': mse_scores,
        'mae_scores': mae_scores
    }
    
    return {
        'model': final_model,
        'scores': mean_scores,
        'fold_scores': scores,
        'feature_importance': pd.DataFrame({
            'Feature': features,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    }

def plot_et_cv_results(scores, mean_scores):
    """
    Visualize cross-validation results for Extra Trees with a line plot showing R² scores across folds.
    
    Parameters:
        scores (dict): Dictionary containing fold-wise scores
        mean_scores (dict): Dictionary containing mean and std of scores
    """
    plt.figure(figsize=(10, 6))
    
    # Create fold numbers for x-axis
    folds = range(1, len(scores['r2_scores']) + 1)
    
    # Plot R² scores across folds
    plt.plot(folds, scores['r2_scores'], 
            'b--o', label='CV R² Scores')
    
    # Add mean R² line
    plt.axhline(y=mean_scores['mean_r2'], color='r', 
                linestyle='-', label=f'Mean R² Score = {mean_scores["mean_r2"]:.4f}')
    
    # Customize the plot
    plt.title('Cross-Validation R² Scores for Extra Trees Model')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.legend()
    
    # Set y-axis limits with some padding
    max_score = max(scores['r2_scores'])
    min_score = min(scores['r2_scores'])
    padding = (max_score - min_score) * 0.1
    plt.ylim(min_score - padding, max_score + padding)
    
    plt.tight_layout()
    plt.show()

    # Print detailed results
    print("\nCross-validation Results:")
    print(f"R² Score: {mean_scores['mean_r2']:.3f} (±{mean_scores['std_r2']:.3f})")
    print(f"MSE: {mean_scores['mean_mse']:.3f} (±{mean_scores['std_mse']:.3f})")
    print(f"MAE: {mean_scores['mean_mae']:.3f} (±{mean_scores['std_mae']:.3f})")

def et_make_predictions(model, data, features, target_column, log_transformed=False):
    """
    Make predictions using the trained Extra Trees model and visualize actual vs predicted values.
    
    Parameters:
        model: Trained Extra Trees model (from extra_trees_train_model results['model'])
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
    
    plt.title('Actual vs Predicted House Prices (Extra Trees)')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    raw_mean_error = results_df['Error'].mean()
    print("\nPrediction Summary:")
    print(f"Mean Absolute Error: ${abs(results_df['Error']).mean():,.2f}")
    print(f"Mean Error: ${raw_mean_error:,.2f}")
    print(f"Error Standard Deviation: ${results_df['Error'].std():,.2f}")
    
    return results_df

def compare_models(model_results_dict):
    """
    Compare and visualize the performance of multiple models.
    
    Parameters:
        model_results_dict (dict): Dictionary with model names as keys and their prediction results as values
        Example: {
            'XGBoost': xgb_predictions,
            'LightGBM': lgb_predictions,
            'CatBoost': catboost_predictions,
            'Extra Trees': et_predictions,
            'Random Forest': rf_predictions
        }
    """
    # Create comparison dataframe
    comparison = []
    for model_name, results in model_results_dict.items():
        errors = results['Actual'] - results['Predicted']
        r2 = r2_score(results['Actual'], results['Predicted'])
        comparison.append({
            'Model': model_name,
            'MAE': abs(errors).mean(),
            'Mean Error': errors.mean(),
            'Std Dev': errors.std(),
            'R² Score': r2
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('MAE')
    
    # Print formatted comparison
    print("\nModel Performance Comparison:")
    print("============================")
    for _, row in comparison_df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"MAE: ${row['MAE']:,.2f}")
        print(f"Mean Error: ${row['Mean Error']:,.2f}")
        print(f"Std Dev: ${row['Std Dev']:,.2f}")
        print(f"R² Score: {row['R² Score']:.4f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot MAE comparison
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e74c3c']
    bars = ax1.bar(comparison_df['Model'], comparison_df['MAE'], color=colors)
    ax1.set_title('Mean Absolute Error by Model')
    ax1.set_ylabel('Mean Absolute Error ($)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom')
    
    # Plot R² Score line graph
    ax2.plot(comparison_df['Model'], comparison_df['R² Score'], 
             'bo-', linewidth=2, markersize=8)
    ax2.set_title('Model Performance Comparison (R² Score)')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('R² Score')
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on line plot
    for x, y in zip(comparison_df['Model'], comparison_df['R² Score']):
        ax2.text(x, y, f'{y:.4f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df


