import matlab.engine
from flame.mode.horizontal.trainer import Trainer
from flame.config import Config
import argparse
import io
import time
import logging

# class PrinterTrainer(Trainer):

#     def __init__(self, config: Config) -> None:
#         self.config = config
#         self.filename = self.config.hyperparameters.filename
#         self.printer_number = self.config.hyperparameters.printer_number
#         self.feedrate = self.config.hyperparameters.feedrate
#         self.acceleration = self.config.hyperparameters.acceleration

#     def _send_weights(self, tag: str) -> None:
#         # logger.debug("calling _send_weights")
#         channel = self.cm.get_by_tag(tag)
#         if not channel:
#             # logger.debug(f"[_send_weights] channel not found with {tag}")
#             return

#         # this call waits for at least one peer to join this channel
#         channel.await_join()

#         # self._update_weights()

#         # delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

#         # delta_weights = self.privacy.apply_dp_fn(delta_weights)

#         msg = {
#             "e_rms" : self.e_rms,
#             "filename" : self.filename,
#             "printer_number" : self.printer_number,
#             "feedrate" : self.feedrate,
#             "acceleration" : self.acceleration,
#         }
#         channel.send(self.aggregator_id, msg)
#         # logger.debug("sending weights done")
    
#     def _fetch_weights(self, tag: str) -> None:
#         # logger.debug("calling _fetch_weights")

#         self.fetch_success = False
#         channel = self.cm.get_by_tag(tag)
#         if not channel:
#             # logger.debug(f"channel not found with tag {tag}")
#             return

#         # this call waits for at least one peer joins this channel
#         channel.await_join()

#         msg, _ = channel.recv(self.aggregator_id)

#         if "feedrate" in msg: # speed
#             self.feedrate = msg["feedrate"]
        
#         if "acceleration" in msg:
#             self.acceleration = msg["acceleration"]

#         # if MessageType.WEIGHTS in msg:
#         #     # logger.debug("received model weights")
#         #     self.weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
#         #     self._update_model()

#         # if MessageType.EOT in msg:
#         #     self._work_done = msg[MessageType.EOT]

#         # if MessageType.ROUND in msg:
#         #     self._round = msg[MessageType.ROUND]

#         self.fetch_success = True
#         # logger.debug(f"work_done: {self._work_done}, round: {self._round}")

#     def train(self):
#         # Start MATLAB engine
#         eng = matlab.engine.start_matlab()
#         eng.addpath('.')  # Add current directory to MATLAB path

#         # Create StringIO objects to capture MATLAB output
#         out = io.StringIO()
#         err = io.StringIO()

#         # Set variables directly in the base workspace
#         print("Setting variables in base workspace...")
#         eng.eval(f"filename_gval = '{self.filename}';", nargout=0)
#         eng.eval(f"feedrate_gval = {self.feedrate};", nargout=0)  
#         eng.eval(f"acceleration_gval = {self.acceleration};", nargout=0)
        
#         # Run the script
#         print("Processing GCode data...")
#         eng.eval("run('Interpret_GCode_RemoveZeroing.m')", nargout=0, stdout=out, stderr=err)
#         print(out.getvalue())
        
#         # Step 3: Run MATLAB simulation and capture the e_rms return value
#         print("\nRunning simulation...")
#         self.e_rms = eng.Simulate_Output_CISCO(self.filename, self.printer_number, self.feedrate, self.acceleration, stdout=out, stderr=err)

#         # Print MATLAB output from simulation
#         print(out.getvalue())
        
#         # Print the e_rms value
#         print(f"\nRMS of the contour error: {self.e_rms} (mm)")

#         # Keep display open for viewing
#         # print("\nKeeping display open for 30 seconds. Press Ctrl+C to exit early.")
#         # time.sleep(30)

#         # Close MATLAB engine
#         eng.quit()


def process_gcode_and_simulate(filename, printer_number, feedrate, acceleration):
    print("Hello")
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    eng.addpath('.')  # Add current directory to MATLAB path

    # Create StringIO objects to capture MATLAB output
    out = io.StringIO()
    err = io.StringIO()

    # Set variables directly in the base workspace
    print("Setting variables in base workspace...")
    eng.eval(f"filename_gval = '{filename}';", nargout=0)
    eng.eval(f"feedrate_gval = {feedrate};", nargout=0)  
    eng.eval(f"acceleration_gval = {acceleration};", nargout=0)
    
    # Run the script
    print("Processing GCode data...")
    eng.eval("run('Interpret_GCode_RemoveZeroing.m')", nargout=0, stdout=out, stderr=err)
    print(out.getvalue())
    
    # Step 3: Run MATLAB simulation and capture the e_rms return value
    print("\nRunning simulation...")
    e_rms = eng.Simulate_Output_CISCO(filename, printer_number, feedrate, acceleration, stdout=out, stderr=err)

    # Print MATLAB output from simulation
    print(out.getvalue())
    
    # Print the e_rms value
    print(f"\nRMS of the contour error: {e_rms} (mm)")

    # Keep display open for viewing
    print("\nKeeping display open for 30 seconds. Press Ctrl+C to exit early.")
    time.sleep(30)

    # Close MATLAB engine
    eng.quit()
    
    return e_rms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Printer Simulation Display")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    filename = config.hyperparameters.filename
    printer_number = config.hyperparameters.printer_number
    feedrate = config.hyperparameters.feedrate
    acceleration = config.hyperparameters.acceleration
    e_rms = process_gcode_and_simulate(filename, printer_number, feedrate, acceleration)
    # Now e_rms is available for further processing in Python
    print(f"Final contour error RMS: {e_rms} mm")

    # trainer = PrinterTrainer(config=config)
    # trainer.compose()
    # trainer.run()



