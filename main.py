from madzik.loader import Loader
from madzik.training import PlumeTrainer
from madzik.processing import PLUMER
from paths import *
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

loader = Loader(DATA_PATH)
trainer = PlumeTrainer(PLUMER({}), loader, "mse", "adam", {
    "test_size": 0.2,
    "validation_size": 0.2,
    "batch_size": 32,
    "epochs": 10,
    "save_path": MODEL_OUTPUT_PATH + dt.datetime.now().strftime("%Y%m%d%H%M%S") + ".h5",
    "load_path": MODEL_OUTPUT_PATH + CURRENT_MODEL_NAME
})
print(trainer.train())
trainer.save()
print(trainer.evaluate())
