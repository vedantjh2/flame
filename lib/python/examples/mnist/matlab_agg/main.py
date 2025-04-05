import logging
from flame.mode.message import MessageType
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.top_aggregator import TopAggregator
from datetime import datetime
import random

# from collections import defaultdict

logger = logging.getLogger(__name__)
E_RMS = "e_rms"
PRINT_TIME = "time"

class PrinterAggregator(TopAggregator):
    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None # set this to weishi's model

        self.dataset: Dataset = None

        self.default_feedrate = self.config.hyperparameters.feedrate
        self.default_acc = self.config.hyperparameters.acceleration

        self.trainer_metrics = {}

        self.trainer_settings = {}

        self.model = None
        self.dataset: Dataset = None

        self.device = None
        self.test_loader = None

    def initialize(self):
        """Initialize role."""
        pass
        

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        # receive local model parameters from trainers
        for msg, metadata in channel.recv_fifo(channel.ends()):
            end, timestamp = metadata
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            logger.debug(f"received data from {end}")
            channel.set_end_property(end, "round_end_time", (round, timestamp))

            e_rms = msg[E_RMS] if E_RMS in msg else None 
            print_time = msg[PRINT_TIME] if PRINT_TIME in msg else None

            self.trainer_metrics[end] = {
                E_RMS : e_rms,
                PRINT_TIME : print_time,
            }


        # logger.debug(f"received {len(self.cache)} trainer updates in cache")


    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        self.dist_tag = tag
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model

        selected_ends = channel.ends()

        # send out global model parameters to trainers
        for end in selected_ends:
            logger.debug(f"sending weights to {end}")
            channel.send(
                end,
                {
                    MessageType.ROUND: self._round,
                    "feedrate": self.trainer_settings[end]["feedrate"] if end  in self.trainer_settings else self.default_feedrate,
                    "acceleration": self.trainer_settings[end]["acceleration"]if end  in self.trainer_settings else self.default_acc,
                },
            )
            # register round start time on each end for round duration measurement.
            channel.set_end_property(
                end, "round_end_time", (round, datetime.now())
            )

        
    def load_data(self) -> None:
        """Load a test dataset."""
        # Implement this if loading data is needed in aggregator
        pass

    def train(self) -> None:
        """Train a model."""
        # Implement this if training is needed in aggregator
        pass


    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        # Implement this if testing is needed in aggregator
        # pass
        for end in self.trainer_metrics:
            err = self.trainer_metrics[end][E_RMS]
            print_time = self.trainer_metrics[end][PRINT_TIME]
            # curr_feedrate, curr_acceleration = self.model(err, print_time) # use this when implementing weishi's model
            curr_feedrate, curr_acceleration = random.choice(range(75, 151, 5)), random.choice(range(5000, 12000, 1000))
            if end not in self.trainer_settings:
                self.trainer_settings[end] = {}
            self.trainer_settings[end]["feedrate"] = curr_feedrate
            self.trainer_settings[end]["acceleration"] = curr_acceleration



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PrinterAggregator(config)
    a.compose()
    a.run()