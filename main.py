# from madzik.loader import Loader
# from madzik.training import PlumeTrainer
import time
from madzik.processing import PLUMER, GASSIAN
from paths import *
import matplotlib.pyplot as plt
import numpy as np
# import datetime as dt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# loader = Loader(DATA_PATH)
# trainer = PlumeTrainer(PLUMER({}), loader, "mse", "adam", {
#     "test_size": 0.2,
#     "validation_size": 0.2,
#     "batch_size": 64,
#     "epochs": 20,
#     "save_path": MODEL_OUTPUT_PATH + dt.datetime.now().strftime("%Y%m%d%H%M%S") + ".weights.h5",
#     "load_path": MODEL_OUTPUT_PATH + CURRENT_MODEL_NAME
# })
# print(trainer.train())
# trainer.save()
# print(trainer.evaluate())
gas = GASSIAN({})
start = time.perf_counter_ns()
print(gas.parse_whole_image(np.zeros((512, 512, 16))))
print(f"Time taken: {(time.perf_counter_ns() - start)/1_000_000}ms")
