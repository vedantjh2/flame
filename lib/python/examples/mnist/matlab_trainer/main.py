import matlab.engine
from flame.channel import VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.mode.message import MessageType
from flame.mode.horizontal.trainer import Trainer
from flame.config import Config
import argparse
import io
import time
import logging

logger = logging.getLogger(__name__)

class PrinterTrainer(Trainer):

    def __init__(self, config: Config) -> None:
        self.config = config
        self.filename = self.config.hyperparameters.filename
        self.printer_number = self.config.hyperparameters.printer_number
        self.feedrate = self.config.hyperparameters.feedrate
        self.acceleration = self.config.hyperparameters.acceleration
        self.model = None
        self.dataset = None
        self.dataset_size = 0

    def initialize(self) -> None:
        pass

    def load_data(self) -> None:
        """Load a test dataset."""
        # Implement this if loading data is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        # Implement this if testing is needed in aggregator
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()
        eng.addpath('.')  # Add current directory to MATLAB path

        # Create StringIO objects to capture MATLAB output
        out = io.StringIO()
        err = io.StringIO()

        # Set variables directly in the base workspace
        print("Setting variables in base workspace...")
        eng.eval(f"filename_gval = '{self.filename}';", nargout=0)
        eng.eval(f"feedrate_gval = {self.feedrate};", nargout=0)  
        eng.eval(f"acceleration_gval = {self.acceleration};", nargout=0)
        
        # Run the script
        print("Processing GCode data...")
        eng.eval("run('Interpret_GCode_RemoveZeroing.m')", nargout=0, stdout=out, stderr=err)
        print(out.getvalue())

        out = io.StringIO()
        err = io.StringIO()
        
        # Step 3: Run MATLAB simulation and capture the e_rms return value
        print("\nRunning simulation...")
        self.e_rms, self.time = eng.Simulate_Output_CISCO(self.filename, self.printer_number, self.feedrate, self.acceleration, nargout=2, stdout=out, stderr=err)
        # Print MATLAB output from simulation
        print(out.getvalue())
        
        # Print the e_rms value
        print(f"\nRMS of the contour error: {self.e_rms} (mm)")
        print(f"\nTime to print: {self.time} (s)")

        # Keep display open for viewing
        # print("\nKeeping display open for 30 seconds. Press Ctrl+C to exit early.")
        # time.sleep(30)

        # Close MATLAB engine
        eng.quit()

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        logger.debug("calling _fetch_weights")

        self.fetch_success = False
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            # we don't want to keep calling this too fast
            # so let's sleep 1 second
            time.sleep(1)
            return

        # this call waits for at least one peer joins this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_RECV)
        msg, _ = channel.recv(end)

        if not msg:
            logger.debug("no message received")
            if self._work_done:
                # when the work is done, we cancel continue condition
                # (i.e., we set fetch_success to True)
                self.fetch_success = True
            # we don't want to keep calling this too fast
            # so let's sleep 1 second
            time.sleep(1)
            return

        if "feedrate" in msg: # speed
            self.feedrate = msg["feedrate"]
            print(f"received new feedrate: {self.feedrate}")
        
        if "acceleration" in msg:
            self.acceleration = msg["acceleration"]
            print(f"received new acceleration: {self.acceleration}")

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        self.fetch_success = True
        logger.debug(f"work_done: {self._work_done}, round: {self._round}") 



    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        msg = {
            "e_rms" : self.e_rms,
            # "filename" : self.filename,
            # "printer_number" : self.printer_number,
            # "feedrate" : self.feedrate,
            # "acceleration" : self.acceleration,
            "time" : self.time,
        }
        channel.send(end, msg)
        logger.debug("sending weights done")
       

    def train(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Printer Simulation Display")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    trainer = PrinterTrainer(config=config)
    trainer.compose()
    trainer.run()



