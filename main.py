from madzik.loader import Loader, JSONLogger, DataSetLoader
from madzik.training import PlumeTrainer
from madzik.validator import Validator
import time
from madzik.processing import PLUMER, GASSIAN
from paths import *
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
import json

# print("Starting training")
# loader = Loader(DATA_PATH)
# plumer = PLUMER({})
# trainer = PlumeTrainer(plumer, loader, "mse", "adam", {
#     "test_size": 0.2,
#     "validation_size": 0.2,
#     "batch_size": 64,
#     "epochs": 20,
#     "save_path": MODEL_OUTPUT_PATH + dt.datetime.now().strftime("%Y%m%d%H%M%S") + ".weights.h5",
#     "load_path": MODEL_OUTPUT_PATH + CURRENT_MODEL_NAME,
#     "callbacks": [JSONLogger("training_logs.json")]
# })
# print("Trainer started")
# print(trainer.train())
# trainer.save()

# print("Training done, loading model")

plumer = PLUMER({})
plumer.model.load_weights(MODEL_OUTPUT_PATH + CURRENT_MODEL_NAME)

print("Starting validation")
val_loader = Loader(VALIDATION_PATH)
val_ds_loader = DataSetLoader(
    val_loader.load_csv_to_id("test.csv"), 32, val_loader)

validator = Validator(val_ds_loader, plumer, {})
evaluation = validator.evaluate()
print(evaluation)
json_eval = {}
for key, value in evaluation.items():
    if isinstance(value, dict):
        for k, v in value.items():
            json_eval[f"tt_{k}"] = float(v)
    else:
        json_eval[key] = float(value)
json_eval["model"] = CURRENT_MODEL_NAME
with open(f"eval_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}_{CURRENT_MODEL_NAME.split('.')[0]}.json", "w") as f:
    json.dump(json_eval, f)

# gas = GASSIAN({})
# start = time.perf_counter_ns()
# print(gas.parse_whole_image(np.zeros((512, 512, 16))))
# print(f"Time taken: {(time.perf_counter_ns() - start)/1_000_000}ms")
