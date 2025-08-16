"""Module with classes and functions for hyper-tuning."""

import json
import pickle
import tensorflow as tf
from tensorflow.keras import layers as L, models as M, callbacks as C
import keras_tuner as kt
from typing import Callable, Optional, Dict, Any, List, Tuple
from pathlib import Path
import warnings

def summarize_best_N_models(num_models: int = 5, tuner: kt.Hyperband = None, metrics: List[str] = None):
    """Summarizes the best N models."""
    if tuner is None:
        raise ValueError("Tuner is required to summarize the best models.")
    
    if metrics is None:
        metrics = ["val_auc_pr", "val_auc_roc", "val_accuracy", "val_precision", "val_recall"]
    
    best_models = tuner.get_best_models(num_models=num_models)
    best_trials = tuner.oracle.get_best_trials(num_trials=num_models)
    best_hps = tuner.get_best_hyperparameters(num_trials=num_models)

    for i, (m, hp, t) in enumerate(zip(best_models, best_hps, best_trials), 1):
        print(f"\n=== Top {i} ===")
        print(f"Scores:")
        print("="*10)
        for metric in metrics:
            print(f"{metric}: {t.metrics.get_best_value(metric)}")
        print("="*10)
        for param in hp.values:
            print(f"{param}: {hp.values[param]}")
        m.summary()


class MetaHyperModel(kt.HyperModel):
    def __init__(self, model_name: str, build_model_func: Callable, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.build_model_func = build_model_func

    def build(self, hp: kt.HyperParameters):
        """Builds the actual model ready for hyper-tuning.

        NOTE: This method passes the hyperparameters to the build_model_func
            and forwards any input arguments too.

        Args:
            hp (kt.HyperParameters): The hyperparameters.

        Returns:
            M.Model: The model.
        """
        return self.build_model_func(hp, self.model_name, **self.kwargs)


class ModelLoader:
    """Class for saving and loading HyperModels with tuning results."""
    
    def __init__(
        self,
        meta_model: MetaHyperModel,
        results_dir: Path,
        random_state: int = 42,
        objective: kt.Objective = kt.Objective("val_auc_pr", direction="max"),
        max_epochs: int = 70,
        factor: int = 3,
    ):
        """Initialize ModelLoader."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.meta_model = meta_model
        self.random_state = random_state
        self.keras_model = None
        self.keras_model_history = None

        self.tuner: kt.Hyperband = kt.Hyperband(
            self.meta_model,
            objective=objective,
            max_epochs=max_epochs, factor=factor, seed=self.random_state,
            directory="tune", project_name=self.meta_model.model_name
        )

        # Reload in case there was a previous search.
        try:
            self.tuner.reload()
        except Exception as e:
            warnings.warn(f"No previous search found for {self.meta_model.model_name}")
    
    def get_tuner(self) -> kt.Hyperband:
        """Get the tuner."""
        return self.tuner
    
    def get_best_model(self) -> M.Model:
        """Get the best model."""
        return self.tuner.get_best_models(1)[0]
    
    def get_best_hyperparameters(self) -> kt.HyperParameters:
        """Get the best hyperparameters."""
        return self.tuner.get_best_hyperparameters(1)[0]
    
    def get_best_model(self) -> M.Model:
        """Get the best model trained with the tuner."""
        best_hp = self.get_best_hyperparameters()
        return self.tuner.hypermodel.build(best_hp)
    
    def get_best_trials(self, num_trials: int = 1) -> List:
        """Get the best trials."""
        return self.tuner.oracle.get_best_trials(num_trials=num_trials)

    def get_best_results(self, metrics: List[str] = ["val_auc_pr", "val_auc_roc", "val_accuracy", "val_precision", "val_recall"]) -> List[Dict[str, Any]]:
        """Get the best results."""
        best_trial = self.get_best_trials(num_trials=1)
        
        return {
            metric: best_trial[0].metrics.get_best_value(metric)
            for metric in metrics
        }
    
    def tune_and_train(self,X_intput, y, X_val, y_val, class_weight=None, epochs=100, batch_size=32, callbacks=None, verbose=1) -> C.History:
        """Tunes and trains the model."""
        self.tuner.search(X_intput, y,
                    validation_data=(X_val, y_val),
                    class_weight=class_weight,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=verbose
                )
        model = self.tuner.get_best_models(1)[0]
        self.keras_model_history = model.fit(
            X_intput, y, 
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose,
        )
        self.keras_model = self.keras_model_history.model
        return self.keras_model_history

    def train_with_fixed_params(self, **kwargs):
        """Trains the model with fixed parameters.
        
        Args:
            **kwargs: HyperParameters to train the model with.

        Returns:
            M.Model: trained model
        """
        hp_fixed = kt.HyperParameters()

        for key, value in kwargs.items():
            hp_fixed.Fixed(key, value)
        
        self.tuner.hypermodel.build(hp_fixed) 
    
    def save_keras_model_to_disk(self):
        """Saves the model to disk."""
        model_path = self.results_dir / "best_model.keras"        
        self.keras_model.save(model_path)
    
    def load_keras_model_from_disk(self) -> Tuple[M.Model, Dict[str, Any]]:
        """Loads model and history from disk uwing the old logic."""
        model_path = self.results_dir / "best_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")
            
        self.keras_model = tf.keras.models.load_model(model_path)
        
        history = self.load_results_from_disk().get("training_params", {}).get("history", {})
            
        print("Loaded model from disk.")
        return self.keras_model, history

    def save_results_to_disk(self, results: Dict[str, Any]):
        """Saves the results to disk."""
        results_path = self.results_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
    
    def load_results_from_disk(self) -> Dict[str, Any]:
        """Loads the results from disk."""
        results_path = self.results_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"No results.json found at {results_path}")
        
        with open(results_path) as f:
            return json.load(f)

    def save_scalers_to_disk(self, scalers: Dict[str, Any]):
        """Saves the scalers to disk."""
        scalers_path = self.results_dir / "scalers.pkl"
        with open(scalers_path, "wb") as f:
            pickle.dump(scalers, f)
    
    def load_scalers_from_disk(self) -> Dict[str, Any]:
        """Loads the scalers from disk."""
        scalers_path = self.results_dir / "scalers.pkl"
        with open(scalers_path, "rb") as f:
            return pickle.load(f)
