import optuna
from madzik.loader import DataSetLoader, Loader
from madzik.processing import PLUMER, GASSIAN
import tensorflow as tf
from sklearn.model_selection import train_test_split


class PlumeTrainer:
    def __init__(self, model: PLUMER, data_loader: Loader, loss_function, optimizer, opts: dict | None = None):
        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        if opts is not None:
            self.test_split = opts["test_size"]
            self.validation_split = opts["validation_size"]
            self.batch_size = opts["batch_size"]
            self.epochs = opts["epochs"]
            self.save_path = opts["save_path"]
            self.load_path = opts["load_path"]
        else:
            self.test_split = 0.2
            self.validation_split = 0.2
            self.batch_size = 32
            self.epochs = 10
            self.save_path = ""
            self.load_path = ""

    def train(self):
        """Train the model"""
        test_subjects = self.data_loader._load_csv_to_id()
        train_subjects, val_subjects = train_test_split(
            test_subjects, test_size=self.test_split)
        train_data = DataSetLoader(
            train_subjects, self.batch_size, self.data_loader)
        val_data = DataSetLoader(
            val_subjects, self.batch_size, self.data_loader)
        data = self.model.model.fit(train_data, epochs=self.epochs,
                                    validation_data=val_data)
        return data

    def evaluate(self):
        """Evaluate the model"""
        test_subjects = self.data_loader._load_csv_to_id()
        test_data = DataSetLoader(
            test_subjects, self.batch_size, self.data_loader)
        return self.model.model.evaluate(test_data)

    def save(self):
        self.model.model.save_weights(self.save_path)

    def load(self):
        self.model.model.load_weights(self.load_path)


class GassianTrainer:
    def __init__(self, model: GASSIAN, data_loader: Loader, loss_function, optimizer, opts: dict | None = None):
        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        if opts is not None:
            self.test_split = opts["test_size"]
            self.validation_split = opts["validation_size"]
            self.batch_size = opts["batch_size"]
            self.epochs = opts["epochs"]
            self.save_path = opts["save_path"]
            self.load_path = opts["load_path"]
        else:
            self.test_split = 0.2
            self.validation_split = 0.2
            self.batch_size = 32
            self.epochs = 10
            self.save_path = ""
            self.load_path = ""

    def train(self):
        """Train the model"""
        test_subjects = self.data_loader._load_csv_to_id()
        train_subjects, val_subjects = train_test_split(
            test_subjects, test_size=self.test_split)
        train_data = DataSetLoader(
            train_subjects, self.batch_size, self.data_loader)
        val_data = DataSetLoader(
            val_subjects, self.batch_size, self.data_loader)
        data = self.model.model.fit(train_data, epochs=self.epochs,
                                    validation_data=val_data)
        return data

    def evaluate(self):
        """Evaluate the model"""
        test_subjects = self.data_loader._load_csv_to_id()
        test_data = DataSetLoader(
            test_subjects, self.batch_size, self.data_loader)
        return self.model.model.evaluate(test_data)

    def save(self):
        self.model.model.save_weights(self.save_path)

    def load(self):
        self.model.model.load_weights(self.load_path)


class HyperparameterOptimizer:
    def __init__(self, model, data_loader, loss_function, optimizer):
        self.model = model
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer

    def objective(self, trial):
        pass

    def optimize(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=100)
        return study.best_params

    def loss_function(self, y_true, y_pred):
        pass

    def save(self):
        pass

    def load(self):
        pass


class GassianOptunaTrainer:
    def __init__(self, model: GASSIAN, data_loader: Loader, opts: dict | None = None):
        self.model = model
        self.data_loader = data_loader
        self.opts = opts

    def train(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.opts["n_trials"])

    def objective(self, trial: optuna.trial.Trial):
        trialopts = {
            "multipliers": [
                trial.suggest_float("multiplier1", 0.1, 1.0),
                trial.suggest_float("multiplier2", 0.1, 1.0),
                trial.suggest_float("multiplier3", 0.1, 1.0),
                trial.suggest_float("multiplier4", 0.1, 1.0),
                trial.suggest_float("multiplier5", 0.1, 1.0),
                trial.suggest_float("multiplier6", 0.1, 1.0),
                trial.suggest_float("multiplier7", 0.1, 1.0),
                trial.suggest_float("multiplier8", 0.1, 1.0),
                trial.suggest_float("multiplier9", 0.1, 1.0),
                trial.suggest_float("multiplier10", 0.1, 1.0),
                trial.suggest_float("multiplier11", 0.1, 1.0),
                trial.suggest_float("multiplier12", 0.1, 1.0),
                trial.suggest_float("multiplier13", 0.1, 1.0),
                trial.suggest_float("multiplier14", 0.1, 1.0),
                trial.suggest_float("multiplier15", 0.1, 1.0),
            ],
            "biases": [
                trial.suggest_float("bias1", -100.0, 100.0),
                trial.suggest_float("bias2", -100.0, 100.0),
                trial.suggest_float("bias3", -100.0, 100.0),
                trial.suggest_float("bias4", -100.0, 100.0),
                trial.suggest_float("bias5", -100.0, 100.0),
                trial.suggest_float("bias6", -100.0, 100.0),
                trial.suggest_float("bias7", -100.0, 100.0),
                trial.suggest_float("bias8", -100.0, 100.0),
                trial.suggest_float("bias9", -100.0, 100.0),
                trial.suggest_float("bias10", -100.0, 100.0),
                trial.suggest_float("bias11", -100.0, 100.0),
                trial.suggest_float("bias12", -100.0, 100.0),
                trial.suggest_float("bias13", -100.0, 100.0),
                trial.suggest_float("bias14", -100.0, 100.0),
                trial.suggest_float("bias15", -100.0, 100.0),
            ],
            "weights": [
                trial.suggest_float("weight1", 0.0, 1.0),
                trial.suggest_float("weight2", 0.0, 1.0),
                trial.suggest_float("weight3", 0.0, 1.0),
                trial.suggest_float("weight4", 0.0, 1.0),
                trial.suggest_float("weight5", 0.0, 1.0),
                trial.suggest_float("weight6", 0.0, 1.0),
                trial.suggest_float("weight7", 0.0, 1.0),
                trial.suggest_float("weight8", 0.0, 1.0),
                trial.suggest_float("weight9", 0.0, 1.0),
                trial.suggest_float("weight10", 0.0, 1.0),
                trial.suggest_float("weight11", 0.0, 1.0),
                trial.suggest_float("weight12", 0.0, 1.0),
                trial.suggest_float("weight13", 0.0, 1.0),
                trial.suggest_float("weight14", 0.0, 1.0),
                trial.suggest_float("weight15", 0.0, 1.0),
            ]
        }

        GASSIAN = GASSIAN(self.opts)
