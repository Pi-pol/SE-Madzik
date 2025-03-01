import os
import numpy as np
import rasterio
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from typing import Generator
import json


class Loader:
    def __init__(self, path: str) -> None:
        self.data_folder_path = path

    def load_folders(self):
        for folder in os.listdir(self.data_folder_path):
            if os.path.isdir(os.path.join(self.data_folder_path, folder)):
                output = self._load_folder(
                    os.path.join(self.data_folder_path, folder))
                yield output
        raise StopIteration

    def load_csv(self, name: str | None = None) -> list[dict[str, str]]:
        """ if name is None, it will load the first csv file in the folder """
        for file in os.listdir(self.data_folder_path):
            if name is None and file.endswith(".csv"):
                return self._parse_csv(os.path.join(self.data_folder_path, file))
            elif file == name:
                return self._parse_csv(os.path.join(self.data_folder_path, file))

    def _load_folder(self, folder) -> dict[str, np.ndarray]:
        output = {
            "mag1c": self._parse_tif(os.path.join(folder, "mag1c.tif")),
            "label_rgba": self._parse_tif(os.path.join(folder, "label_rgba.tif")),
            "label_binary": self._parse_tif(os.path.join(folder, "labelbinary.tif")),
            "460nm": self._parse_tif(os.path.join(folder, "TOA_AVIRIS_460nm.tif")),
            "550nm": self._parse_tif(os.path.join(folder, "TOA_AVIRIS_550nm.tif")),
            "640nm": self._parse_tif(os.path.join(folder, "TOA_AVIRIS_640nm.tif")),
            "2004nm": self._parse_tif(os.path.join(folder, "TOA_AVIRIS_2004nm.tif")),
            "2109nm": self._parse_tif(os.path.join(folder, "TOA_AVIRIS_2109nm.tif")),
            "2310nm": self._parse_tif(os.path.join(folder, "TOA_AVIRIS_2310nm.tif")),
            "2350nm": self._parse_tif(os.path.join(folder, "TOA_AVIRIS_2350nm.tif")),
            "2360nm": self._parse_tif(os.path.join(folder, "TOA_AVIRIS_2360nm.tif")),
            "WV3_SWIR1": self._parse_tif(os.path.join(folder, "TOA_WV3_SWIR1.tif")),
            "WV3_SWIR2": self._parse_tif(os.path.join(folder, "TOA_WV3_SWIR2.tif")),
            "WV3_SWIR3": self._parse_tif(os.path.join(folder, "TOA_WV3_SWIR3.tif")),
            "WV3_SWIR4": self._parse_tif(os.path.join(folder, "TOA_WV3_SWIR4.tif")),
            "WV3_SWIR5": self._parse_tif(os.path.join(folder, "TOA_WV3_SWIR5.tif")),
            "WV3_SWIR6": self._parse_tif(os.path.join(folder, "TOA_WV3_SWIR6.tif")),
            "WV3_SWIR7": self._parse_tif(os.path.join(folder, "TOA_WV3_SWIR7.tif")),
            "WV3_SWIR8": self._parse_tif(os.path.join(folder, "TOA_WV3_SWIR8.tif")),
            "weight": self._parse_tif(os.path.join(folder, "weight_mag1c.tif")),
        }
        return output

    def load_folder_by_id(self, id) -> dict[str, np.ndarray]:
        folder = os.path.join(self.data_folder_path, str(id))
        return self._load_folder(folder)

    def _parse_tif(self, file) -> np.ndarray:
        with rasterio.open(file) as img:
            return img.read()

    def _parse_csv(self, file) -> list[dict[str, str]]:
        data = []
        with open(file, "r") as f:
            columns = f.readline().strip().split(",")
            for line in f:
                values = line.strip().split(",")
                data.append(dict(zip(columns, values)))
        return data

    def load_csv_to_id(self, name: str | None = None) -> list[tuple[str, bool]]:
        if name is None:
            csv = self.load_csv("train.csv")
        else:
            csv = self.load_csv(name)
        output = [(row["id"], row["has_plume"] == "True") for row in csv]
        return output

    def combine_into_one_ndarray(self, data) -> np.ndarray:
        data.pop("weight")
        data.pop("label_rgba")
        data.pop("label_binary")
        data.pop("mag1c")

        # The data is already (1, 512, 512), just need to properly stack
        arrays = [data[key].squeeze() for key in sorted(data.keys())]
        stacked = np.stack(arrays, axis=-1)
        return stacked.astype(np.float32)

    def create_color_composite(self, data) -> np.ndarray:
        return np.stack((data["640nm"], data["550nm"], data["460nm"]), axis=-1)


class DataSetLoader(keras.utils.PyDataset):
    def __init__(self, ids: list | None = None, batch_size=32, data_loader: Loader = None, **kwargs) -> None:
        self.data_loader = data_loader
        if ids is None:
            self.labels = self.data_loader.load_csv_to_id()
        else:
            self.labels = [id for id in ids]
        self.labels_original = self.labels.copy()
        self.batch_size = batch_size
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return len(self.labels) // self.batch_size

    def get_batch(self, idx) -> list:
        batch = []
        while len(batch) < self.batch_size:
            label = self.labels.pop(0)
            if os.path.exists(os.path.join(self.data_loader.data_folder_path, label[0])):
                batch.append(label)
        return batch

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        batch = self.get_batch(idx)
        data = [self.data_loader.load_folder_by_id(id) for id, _ in batch]
        x = np.stack([self.data_loader.combine_into_one_ndarray(d)
                     for d in data])
        y = np.array([label for _, label in batch])
        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.labels)
        self.labels = self.labels_original.copy()


class JSONLogger(tf.keras.callbacks.Callback):
    def __init__(self, json_path):
        super().__init__()
        self.json_path = json_path
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch} ended, loss: {logs['loss']}, accuracy: {logs['accuracy']}")
        self.logs.append(
            {"epoch": epoch, "loss": logs["loss"], "accuracy": logs["accuracy"]})
        with open(self.json_path, 'w') as f:
            json.dump(self.logs, f)
