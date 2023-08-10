# Remote-Baby-Monitoring-System

## Project Description 

In today's world parents strugle in monitoring and properly caring their babies with their tight work schedule. Over the years although systems with cameras to watch the baby have been produced there isn't a complete system to completely monitor the baby and to identify his emotions. This project aims to develop such a system by identifying the emotions of the crying of baby using machine learning and by providing features to monitor and interact with baby from remote using a mobile app.

## Features
1. Give an alert in mobile app when baby wakes up using motion detection sensors
2. Gives an aleart when baby is crying by capturing audio and distinguishing it with background noice
3. Indicate the emotion of the crying as hungry, discormfort, tired and belly pain using machine learning techniques
4. Ability to watch the condition of baby through the mobile app using the camera integrated to the system
5. Ability to speak to the baby through the mobile app

## Project Includes
1. Source code of the embedded system
2. Design files for the pcb of the system
3. Trained machine learning model used
4. Data set used for training

## Technologies and Hardware Used
1. ESP-32 development board
2. ESP camera module
3. Tensorflow lite platform to train the model
4. Blynk library to create the mobile app

## Credits
1. donateacry-corpus dataset by gveres - https://github.com/gveres/donateacry-corpus
2. Audio classification using Tensorflow - https://www.tensorflow.org/lite/examples/audio_classification/overview
3. Audio feature extraction - https://devopedia.org/audio-feature-extraction#:~:text=Audio%20feature%20extraction%20is%20a,converting%20digital%20and%20analog%20signals.
4. Deep learning to classify baby crying - https://towardsdatascience.com/deep-learning-for-classifying-audio-of-babies-crying-9a29e057f7ca

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
