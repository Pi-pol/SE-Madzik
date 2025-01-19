from madzik.loader import Loader
from madzik.training import PlumeTrainer
from madzik.processing import PLUMER


class CLI():
    def __init__(self):
        self.loader: Loader | None = None
        self.model: PLUMER | None = None
        self.trainer: PlumeTrainer | None = None

    def run(self):
        while True:
            pass

    def first_start(self):
        print("Welcome to Madzik!")

    def print_help(self):
        print("Commands:")
        print("h|help - print this help message")
        print("q|quit - quit the program")
        print("load [path] - load the model")
        print("load_model [path] - load the model")
        print("predict - predict the model")
        print(
            "addjust_trainging [key:value]- adjust the training hyperparameters")
        print("                  epochs: [int] - number of epochs")
        print("                  batch_size: [int] - batch size")
        print("                  test_size: [float] - test size")
        print("train - train the model")

    def handle_command(self, command: str):
        command = command.strip()
        if command.startswith("h") or command.startswith("help"):
            self.print_help()
        elif command.startswith("q") or command.startswith("quit"):
            exit(0)
        elif command.startswith("load") or command.startswith("load_model"):
            pass
        elif command.startswith("predict"):
            pass
        elif command.startswith("adjust_training"):
            pass
        elif command.startswith("train"):
            pass
        else:
            print("Unknown command")

    def handle_adjust(self):
        pass