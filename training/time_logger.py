import time
from pytorch_lightning.callbacks import Callback


class TrainingTimeLogger(Callback):
    def __init__(self):
        self.train_end_time = None
        self.train_start_time = None

    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        print("Training started.")

    def on_train_end(self, trainer, pl_module):
        self.train_end_time = time.time()
        elapsed_time = self.train_end_time - self.train_start_time
        print(f"Training finished. Elapsed time: {elapsed_time:.2f} seconds")