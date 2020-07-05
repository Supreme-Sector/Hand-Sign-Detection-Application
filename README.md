# Hand-Sign-Detection-Application
The code can be used to construct a Detector object which will detect the hand sign for the letter Y through a camera <br>
Multiple neural networks are connected together in order to analyze images <br>
If the hand sign for Y is detected, then `final_application.py` opens youtube.com <br>
## Usage
Run `training_data_collector.py` and `test_data_collector.py` to collect training and test data <br>
After that run `subnet_trainer.py` to generate a Network pickle <br>
If the accuracy is high enough, save the pickle by giving it a unique name and moving it to the `network_pickles` folder. <br>
Run `ensemble_training_data_collector.py` with the proper initialization of the variables. <br>
Run `intermediate_network_trainer.py`. <br>
Rename and save the Network pickle. <br>
In `y_sign_detection.py` make sure the Detector is instantiated with the appropriate sub and intermediate Network objects. <br>
Run the script and it should return the final accuracy of the Detector. <br>
Below the accuracy, there should also be an message that says "Detector Saved" <br>
If you get a MemoryError, try commenting out the part of the script that evaluates the Detector and run again. <br>
Once you have a final detector pickle, run `final_application.py`. <br>
Whenever you present the Y hand sign at the camera, youtube.com will open up. <br>
## Videos
Part 1: https://youtu.be/GdPtevjUfzo <br>
Part 2: https://youtu.be/nhyEixTzuDg <br>
Part 3: Coming soon ... <br>
