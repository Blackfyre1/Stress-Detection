Audio Based Stress Detection using DAIC-WOZ Dataset

Overview
This machine learning model aims to detect stress levels based on input data. It utilizes various neural network architectures such as Deep Neural Networks (DNN), Convolutional Neural Networks (CNN), and Long Short-Term Memory networks (LSTM) for stress detection.

Model Architecture
The model consists of three different architectures:
Deep Neural Network (DNN)
Layers:
Input Layer (193 features)
Dense Layer (64 units, ReLU activation)
Dense Layer (32 units, ReLU activation)
Output Layer (1 unit, Sigmoid activation)
Convolutional Neural Network (CNN)
Layers:
Conv1D Layer (128 filters, kernel size 8, ReLU activation)
MaxPooling1D Layer (pool size 2)
Conv1D Layer (64 filters, kernel size 8, ReLU activation)
MaxPooling1D Layer (pool size 2)
Flatten Layer
Output Layer (1 unit, Sigmoid activation)
Long Short-Term Memory network (LSTM)
Layers:
LSTM Layer (128 units)
Dense Layer (64 units, ReLU activation)
Dense Layer (32 units, ReLU activation)
Output Layer (1 unit, Sigmoid activation)

Usage Instructions:
Prerequisites:
Python (3.x recommended)
Required libraries: TensorFlow, pandas, numpy, matplotlib, sklearn
Steps:
Set Up Environment:
Ensure Python is installed. If not, download and install Python (3.x) from [Python's official website](https://www.python.org/downloads/).
Install Required Libraries:
Open the terminal or command prompt.
Navigate to the project directory.
Run `pip install -r requirements.txt` to install necessary libraries.
Prepare the Data:
Ensure the dataset is available or use your own dataset.
Modify the file paths in the code (`data_analysis.ipynb` and `stress_detection.ipynb`) to point to your dataset if necessary.
Run Data Analysis:
Open `data_analysis.ipynb` using Jupyter Notebook or any compatible IDE.
Execute each cell sequentially to extract audio features from the test audio file.
Run Stress Detection:
Open `stress_detection.ipynb` using Jupyter Notebook or any compatible IDE.
Ensure that the extracted features from the previous step are accessible to `stress_detection.ipynb`.
Run each cell sequentially in `stress_detection.ipynb` to perform data preprocessing, model training, and evaluation.
Customization and Experimentation:
Tweak hyperparameters, modify architectures, or adjust data preprocessing steps for experimentation and customization.

Notes :
The provided code in the notebooks (`data_analysis.ipynb` and `stress_detection.ipynb`) demonstrates the entire workflow from data preprocessing to model evaluation.
Customization and fine-tuning of hyperparameters may be required for specific use cases.
