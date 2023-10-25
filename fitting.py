import pandas as pd
import numpy as np
import catboost as cb
import optuna
from sklearn.model_selection import train_test_split

def get_objective_Catboost(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Generates an objective function for training with optuna based on the data.

    Args:
        X_train   (Pandas DataFrame)
        y_train      (Pandas Series)
    Returns:
        (Callable): Objective function for optuna
    """
    
    def objective(trial: optuna.Trial) -> float:
        """
        Objective for training with optuna.
        Potential conditional parameters:
        - bagging_temperature (dependency: bootstrap_type = Bayesian)
        - subsample           (dependency: bootstrap_type in (Poisson, Bernoulli, MVS))
        - MVS_reg             (dependency: bootstrap_type = MVS)

        Args:
            trial (Optuna trial): Connection to Optuna to optimize hyperparameters
        Returns:
            Some score to guide optuna
        """
        
        # Hyperparameters
        iterations = trial.suggest_int("iterations", 10, 600)
        early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 2, 50)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        depth = trial.suggest_int("depth", 4, 10)
        l2_leaf_reg = trial.suggest_int("l2_leaf_reg", 1, 100)
        random_strength = trial.suggest_float("random_strength", 0.1, 2.)
        rsm = trial.suggest_float("rsm", 0.1, 1)
        grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
        if grow_policy != "Lossguide":
            sampling_frequency = trial.suggest_categorical("sampling_frequency", ["PerTreeLevel", "PerTree"])
        elif grow_policy == "Lossguide":
            max_leaves = trial.suggest_int("max_leaves", 1, 64)
        else:
            sampling_frequency = "PerTree"
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 20)
        border_count = trial.suggest_int("border_count", 5, 254)
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]) 
        if bootstrap_type == "Bayesian":
            bagging_temperature = trial.suggest_float("bagging_temperature", 0, 10)
        elif bootstrap_type in ("Bernoulli", "MVS"):
            subsample = trial.suggest_float("subsample", 0.1, 1)
        

        params = {
            'iterations': iterations,
            'early_stopping_rounds': early_stopping_rounds,
            'learning_rate': learning_rate,
            'depth': depth,
            'l2_leaf_reg': l2_leaf_reg,
            'random_strength': random_strength,
            'rsm': rsm,
            'grow_policy': grow_policy,
            'min_data_in_leaf': min_data_in_leaf,
            'border_count': border_count,
            'bootstrap_type': bootstrap_type,
            'loss_function': "MAE", #Changed from MAE
            'random_seed': 42,
            'verbose': False,
            'thread_count': 16
        }


        if bootstrap_type == "Bayesian":
            params['bagging_temperature'] = bagging_temperature
        elif bootstrap_type in ("Bernoulli", "MVS"):
            params['subsample'] = subsample

        if grow_policy != "Lossguide":
            params['sampling_frequency'] = sampling_frequency
        elif grow_policy == "Lossguide":
            params['max_leaves'] = max_leaves


        stores_train_catboost = cb.Pool(
            data=X_train,
            label=y_train,
        )
        
        #Does cross validation on the hyperparameters, 10 folds
        result = cb.cv(
            stores_train_catboost,
            params,
            nfold=10,
            return_models=False,
            as_pandas=True,
            logging_level='Silent'
        )

        return np.expm1(result["test-MAE-mean"].iloc[-1])

    return objective


def fit_CatBoost(params: dict, X: pd.DataFrame, y: pd.Series, valid_size: float = 0.3) -> cb.CatBoostRegressor:
    """
    Performs CatBoost on the dataset above given some param dictionary
    which is determined by Optuna.

    Args:
        params      (Dictionary): Parameters determined by optuna (dict format)
        X:    (Pandas DataFrame): Training data
        y:       (Pandas Series): Target 
        cat_features (list[str]): Categorical features
    Returns:
        Trained Catboost model
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, random_state=42)
    eval_set = cb.Pool(
        X_valid,
        y_valid
    )

    mod = cb.CatBoostRegressor(
        **params,
        thread_count=-1,
        loss_function='MAE',#These two are changed from MAE to RMSE
        eval_metric='MAE',
        random_seed=42,
        verbose=False
    )

    mod.fit(X_train, y_train, eval_set=eval_set)

    return mod

def generate_predictions(mod, X_test) -> dict:
    """
    Generates predictions of the appropriate form based on models for each location
    (must have a .predict method) and test data.

    Args:
        mod_a           (Model)
        X_test          (pd.DataFrame)
    Returns:
        (Dictionary): Dataframe of predictions for each location
    """
    predictions = {}

    pred = np.expm1(mod.predict(X_test))
    pred[pred < 1e-3] = 0
    

    predictions = pd.DataFrame({ "pred": pred})

    return predictions