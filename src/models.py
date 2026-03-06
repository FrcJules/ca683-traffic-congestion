"""
Model Training and Evaluation Utilities

This module provides functions for training and evaluating ML models.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import cross_val_score
import time


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    verbose: bool = True
) -> Tuple[Any, float]:
    """
    Train a machine learning model and measure training time.

    Args:
        model: Scikit-learn compatible model
        X_train: Training features
        y_train: Training target
        verbose: Whether to print training info

    Returns:
        Tuple of (trained_model, training_time_seconds)

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=100)
        >>> model, train_time = train_model(model, X_train, y_train)
    """
    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    if verbose:
        print(f"✓ Model trained in {training_time:.2f}s")

    return model, training_time


def evaluate_regression_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a regression model.

    Args:
        model: Trained regression model
        X_test: Test features
        y_test: True test labels
        verbose: Whether to print metrics

    Returns:
        Dictionary of evaluation metrics

    Example:
        >>> metrics = evaluate_regression_model(model, X_test, y_test)
    """
    y_pred = model.predict(X_test)

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mean_actual': y_test.mean(),
        'mean_predicted': y_pred.mean()
    }

    if verbose:
        print("="*50)
        print("REGRESSION MODEL EVALUATION")
        print("="*50)
        print(f"RMSE:  {metrics['rmse']:.2f}")
        print(f"MAE:   {metrics['mae']:.2f}")
        print(f"R²:    {metrics['r2']:.4f}")
        print(f"Mean Actual:     {metrics['mean_actual']:.2f}")
        print(f"Mean Predicted:  {metrics['mean_predicted']:.2f}")
        print("="*50)

    return metrics


def evaluate_classification_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    average: str = 'binary',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a classification model.

    Args:
        model: Trained classification model
        X_test: Test features
        y_test: True test labels
        average: Averaging strategy for multi-class ('binary', 'macro', 'weighted')
        verbose: Whether to print metrics

    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_test, y_pred, average=average, zero_division=0)
    }

    if verbose:
        print("="*50)
        print("CLASSIFICATION MODEL EVALUATION")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print("="*50)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    return metrics


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.

    Args:
        model: Scikit-learn compatible model
        X: Features
        y: Target
        cv: Number of folds
        scoring: Scoring metric
        verbose: Whether to print results

    Returns:
        Dictionary with cross-validation results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    results = {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'all_scores': scores.tolist()
    }

    if verbose:
        if 'neg' in scoring:
            # Convert negative scores back to positive
            print(f"CV Score: {-results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        else:
            print(f"CV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")

    return results


def compare_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = 'regression'
) -> pd.DataFrame:
    """
    Train and compare multiple models.

    Args:
        models: Dictionary of {model_name: model_instance}
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_type: 'regression' or 'classification'

    Returns:
        DataFrame with comparison results

    Example:
        >>> models = {
        >>>     'Random Forest': RandomForestRegressor(),
        >>>     'XGBoost': XGBRegressor(),
        >>>     'LightGBM': LGBMRegressor()
        >>> }
        >>> comparison = compare_models(models, X_train, y_train, X_test, y_test)
    """
    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train
        trained_model, train_time = train_model(model, X_train, y_train, verbose=False)

        # Evaluate
        if model_type == 'regression':
            metrics = evaluate_regression_model(trained_model, X_test, y_test, verbose=False)
            results.append({
                'Model': name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'Training Time (s)': train_time
            })
        else:  # classification
            metrics = evaluate_classification_model(trained_model, X_test, y_test, verbose=False)
            results.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'Training Time (s)': train_time
            })

    # Create comparison DataFrame
    df_comparison = pd.DataFrame(results)

    # Sort by best metric
    if model_type == 'regression':
        df_comparison = df_comparison.sort_values('RMSE')
    else:
        df_comparison = df_comparison.sort_values('F1', ascending=False)

    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(df_comparison.to_string(index=False))
    print("="*70)

    return df_comparison


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract and rank feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance rankings
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance_df.head(top_n)


def predict_with_confidence(
    model: Any,
    X: pd.DataFrame,
    confidence_percentile: float = 95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with confidence intervals (for tree-based ensembles).

    Args:
        model: Trained ensemble model
        X: Features to predict
        confidence_percentile: Percentile for confidence interval

    Returns:
        Tuple of (predictions, lower_bound, upper_bound)
    """
    # For ensemble models, get predictions from individual trees
    if hasattr(model, 'estimators_'):
        # Get predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in model.estimators_])

        # Calculate mean and confidence intervals
        predictions = all_predictions.mean(axis=0)
        lower_bound = np.percentile(all_predictions, (100 - confidence_percentile) / 2, axis=0)
        upper_bound = np.percentile(all_predictions, 100 - (100 - confidence_percentile) / 2, axis=0)

        return predictions, lower_bound, upper_bound
    else:
        # For non-ensemble models, just return predictions
        predictions = model.predict(X)
        return predictions, predictions, predictions
