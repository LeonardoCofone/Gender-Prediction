# Gender Prediction AI  
**Gender classification from face images using deep learning**  

---

## 1 · Introduction  

This repository contains the complete code and documentation for a real-time gender prediction system based on deep learning, developed entirely by **Leonardo Cofone**.  

The project covers:  
- Face detection using MediaPipe  
- Gender classification using a custom-trained TensorFlow CNN model  
- Real-time webcam inference with a friendly graphical user interface (GUI) built with Kivy  
- Image upload and prediction functionality  

The aim is to provide an easy-to-use application for gender prediction with clear visual feedback and fast performance on typical desktop environments.  

---

## 2 · Dataset  

| Feature       | Value                                         |
| ------------- | --------------------------------------------- |
| Origin        | Too see the data visit the project on [kaggle](https://www.kaggle.com/code/zlatan599/gender-prediction/notebook) |
| Size          | Approximate 100k image of different faces  |
| Classes       | 2 classes: Male, Female                        |

### Notes  
- The trained model (`Gender_prediction_final.h5`) is not included but can be downloaded from the [Kaggle project](https://www.kaggle.com/code/zlatan599/gender-prediction/notebook)

---

## 3 · Project Structure  
├── GUI.py                                  # Main application script with GUI and live camera  
├── model/  
│   ├── Gender_prediction_final.h5          # Pre-trained TensorFlow model (not included)  
│   └── best_threshold.txt                  # Classification threshold  
├── assets/  
│   └── LOGO.png                            # Logo image used in loading screen  
├── example.mp4                             # An example of how to use the app   
├── README.md                               # This documentation file  
├── LICENSE.txt                             # The licensa (MIT)  
├── Creation_of_the_model.ipynb             # The notebook used to create the model

---

## Thank you all
> If you have something to report, contact me
---
[Kggle](https://www.kaggle.com/zlatan599)  
[Linkedin](https://www.linkedin.com/in/leonardo-cofone-914228361/)



