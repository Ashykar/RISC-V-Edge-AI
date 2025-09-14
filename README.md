# RISC-V-Edge-AI

This GitHub repository documents the [RISC-V Edge AI with VSDSquadron Pro 10-days Workshop](https://www.vlsisystemdesign.com/riscv_edgeai/) offered by [VSD Corp. Pvt. Ltd.](https://www.vlsisystemdesign.com/about-us/) attended from 05-14 September, 2025.<br><br>

## Table of Contents

| Module # | Topic(s) Covered | Status |
|----------|------------------|--------|
| [Module 1](#1---edge-aI-orientation-&-hardware-primer) | **Edge AI Orientation & Hardware Primer**<br>0. [AI On A Microchip - Edge Computing With VSDSquadron Pro RISC-V Board](#11---aI-on-a-microchip---edge-computing-with-VSDSquadron-pro-RISC---V-board-)<br>1. [Understanding Your RISC-V Board - Prerequisites to AI on 16KB RAM](#12---understanding-your-rISC---V-board---prerequisites-to-aI-on-16KB-rAM) | <img width="94" height="20" alt="image" src="https://github.com/user-attachments/assets/9768417d-9351-4e4d-b293-5d0490a1225c" /> |
| [Module 2]() | **ML Foundations (Regression & Optimization)**<br>2. [Best-Fitting Lines 101 - Getting Started With ML]()<br>3. [Gradient Descent Unlocked - Build Your First AI Model From Scratch]()<br>4. [Visualizing Gradient Descent in Action]()<br>5. [Predicting Startup Profits – AI for Business Decisions]()<br>6. [Degree Up - Fitting Complex Patterns for Edge AI]()<br>7. [From Python to Silicon - Your Model Runs on RISC-V (Need VSDSQ Board)]() | <img width="94" height="20" alt="image" src="https://github.com/user-attachments/assets/9768417d-9351-4e4d-b293-5d0490a1225c" /> |
| [Module 3]() | **From Regression to Classification (KNN->SVM)**<br>8. [From Regression to Classification - Your First Binary AI Model]()<br>9. [Implementing KNN Classifier in Python - Smarter Decision Boundaries]()<br>10. [ From KNN to SVM - Smarter Models for Embedded Boards]()<br>11. [Deploying SVM Models on VSDSquadron PRO Boards - From Python to C]()<br>12. [Handwritten Digit Recognition with SVM - From MNIST to Embedded Boards]()<br>13. [Running MNIST Digit Recognition on the VSDSquadron PRO Board]() | <img width="94" height="20" alt="image" src="https://github.com/user-attachments/assets/9768417d-9351-4e4d-b293-5d0490a1225c" /> |
| [Module 4]() | **Memory-Constrained ML & Quantization**<br>14. [Beating RAM Limits - Quantizing ML Models for Embedded Systems]()<br>15. [Quantization Demystified - Fitting AI Models on Tiny Devices]()<br>16. [ Post-Training Quantization - From 68KB Overflow to MCU-Ready AI]()<br>17. [Fitting AI into 16KB RAM - The Final Embedded ML Optimization (Need VSDSQ Board)]()<br>18. [Regression to Real-Time Recognition - A Complete Embedded ML Recap]() | <img width="94" height="20" alt="image" src="https://github.com/user-attachments/assets/9768417d-9351-4e4d-b293-5d0490a1225c" /> |
| [Module 5]() | **Neural Networks on RISC-V Microcontrollers**<br>19. [From Brain to Code - How Neurons Inspired Artificial Intelligence]()<br>20. [From SVM to Neural Networks - Adding Hidden Layers]()<br>21. [Neural Networks in Action - From Scratch to 98% Accuracy]()<br>22. [Can We Fit a Neural Network on VSDSQ PRO Board - Memory Math Explained]()<br>23. [From VSDSQ Mini to VSDSQ Pro - Real-Time AI Digit Recognition on VSD Boards]() | <img width="94" height="20" alt="image" src="https://github.com/user-attachments/assets/9768417d-9351-4e4d-b293-5d0490a1225c" /> |
| [Module 6]() | **Advanced Quantization & Deployment**<br>24. [Neural Network Implementation Repository]()<br>25. [Training Bit-Quantized Neural Network Implementation with Quantization-Aware Training]()<br>26. [Test Your Quantized Model]()<br>27. [Exporting Bit-Quantized Neural Network to RISC-V]() | <img width="94" height="20" alt="image" src="https://github.com/user-attachments/assets/9768417d-9351-4e4d-b293-5d0490a1225c" /> |
| [Module 7]() | **Capstone & Next Steps**<br>28. [Moving Forward and New Opportunities]() | <img width="94" height="20" alt="image" src="https://github.com/user-attachments/assets/9768417d-9351-4e4d-b293-5d0490a1225c" /> |

## **1. Edge AI Orientation & Hardware Primer**

### **1.0. AI On A Microchip - Edge Computing With VSDSquadron Pro RISC-V Board**

Edge AI involves running artificial intelligence (AI) and machine learning (ML) algorithms directly on devices at the edge of a network, such as smartphones, IoT devices, or smart cameras, rather than sending data to a distant cloud for processing. This on-device processing enables faster, localized responses with lower latency, improved data privacy and security, and reduced bandwidth usage.

**Programming: Logic Vs Learning**
     
| **Traditional logic** | **AI** |
|-----------------------|--------|
| Programmer explicitly defines behavior. | No explicit rules; instead, provide training data. |
| Given the same input, it always produces the same output. | It learns pattern, AI generalizes from examples instead of fixed instructions. |
| Like RISC-V assembly; Low-level, deterministic, tightly controlled. | Iterative, adaptive, sometimes unpredictable like a toddler learning by trial and error |

Learning-based AI uses algorithms to learn patterns and rules from data rather than being explicitly programmed for every scenario, allowing systems to improve their performance and decision-making over time through experience. This approach is centered on machine learning, where models are trained on large datasets to make predictions or classifications on new, unseen data. Key types include supervised learning (using labeled data), unsupervised learning (identifying patterns in unlabeled data), and reinforcement learning (learning from rewards and penalties).  

Challenges on RISC-V (VSD Pro board):

- Unlike Raspberry Pi, which has a full OS and tools, RISC-V is bare-metal, so you need to code closer to the hardware.
- Limited resources but higher efficiency and control.
- Difficult but also the possibility of success.

### **1.1. Understanding Your RISC-V Board - Prerequisites to AI on 16KB RAM**

**VSDSquadron PRO RISC-V development board Block Diagram**<br>
![Block Diagram](https://github.com/user-attachments/assets/2ab52678-8a19-45dd-9f39-071cf00dd47c)<br>
**Overview of VSDSquadron PRO board powered by SiFive**

- On-board 16MHz crystal
- 19 Digital IO pins and 9 PWM pins
- 2 UART and 1 I2C
- Dedicated quad-SPI (QSPI) flash interface
- 32 Mbit Off-Chip (ISSI SPI Flash)
- USB-C type for Program, Debug, and Serial Communication
- Memory for Enhanced Learning: **With a 16KB L1 Instruction Cache and a 16KB Data SRAM scratchpad**, the board allows students to experiment with data processing and efficient instruction handling

**VSDSquadron PRO RISC-V development board**<br>
![VSDSquadron PRO RISC-V development board](https://github.com/user-attachments/assets/0abfa42d-1227-4a4e-b63d-1a6c03a274b8)<br>
**Specifications of the VSDSquadron PRO Board**<br>
![Specifications](https://github.com/user-attachments/assets/8395aeb1-7b4c-44c9-9f90-72196eec1511)<br>

**Getting Started Guide**

1. Board Datasheet: Download the [Board Datasheet](https://www.vlsisystemdesign.com/wp-content/uploads/2024/10/datasheet.pdf) 
2. Installing SiFive Software: To program and debug the RISC-V development board, you will need SiFive Freedom Studio. 
3. Sign into [Google Colab](https://colab.research.google.com/) account
- Open new notebook in drive from File menu

**Prerequisites**
- Python
- C++
- Basic understanding of RISC V
- A lot of Maths 

## **2. ML Foundations (Regression & Optimization)**

### **2.2. Best-Fitting Lines 101 - Getting Started With ML**

In machine learning, several key concepts are central to building predictive models:

**1. Data Points:** These are individual observations or instances in a dataset, each typically consisting of input features (independent variables) and an associated output or target value (dependent variable).

Example: In predicting house prices, a data point could be (Square Footage: 1500, Number of Bedrooms: 3, Actual Price: $300,000).

**2. Prediction and Predicted Values:**
Prediction: The process of using a trained machine learning model to estimate the output or target value for new, unseen input data. The specific output values generated by the model during the prediction process are called the Predicted Values.

Example: After training a house price prediction model, if you input (Square Footage: 1600, Number of Bedrooms: 4), the model might output a predicted price of $320,000. This $320,000 is the predicted value.

**3. Error:** The difference between the actual value of a data point and the value predicted by the model for that same data point. It quantifies how well the model's predictions align with reality.

**Error = Actual Value - Predicted Value**

Example: If the actual price of a house was $300,000, and the model predicted $320,000, the error for that data point is $300,000 - $320,000 = -$20,000.

The primary objective in training most machine learning models is to find the model parameters that minimize the overall error across the entire training dataset. This is often achieved by minimizing a "loss function" (e.g., Mean Squared Error in linear regression), which quantifies the aggregate error.

Example: In linear regression, algorithms like Gradient Descent iteratively adjust the slope and intercept of the line to reduce the sum of squared errors.

**Best Fit:** The "line of best fit" (or "curve of best fit" for non-linear models) is the representation of the trained model that best captures the underlying relationship between the input features and the target variable, as determined by minimizing the error.

Example: In linear regression, the line of best fit is the straight line that minimizes the sum of squared errors between the actual house prices and the prices predicted by the line.<br>
<img width="691" height="547" alt="Best fit" src="https://github.com/user-attachments/assets/e9361a10-35bf-42c6-ba8a-9da52560700d" /><br>
- On a scatter plot, each data point is represented as a distinct mark (e.g., a dot) with its coordinates corresponding to the input features and target value.
- On a graph with a "line of best fit," for any given input on the x-axis, the corresponding point on the line represents the predicted value.
- The vertical distance between a data point and the "line of best fit" represents the error for that specific point.
- Minimizing error means finding the line that, on average, is closest to all the data points, resulting in the smallest vertical distances between the points and the line that visually appears to most accurately represent the trend or relationship within the data, having minimized the distances to the individual data points.

In this graph:
- **Blue dots:** are the data points representing actual values.
- The **red line** is the best fit line, representing the predicted values for given input features.
- The **green dashed line** shows the error for a specific data point, which is the vertical distance between the actual data point and the best-fit line. 

The goal of machine learning is to find the red line that minimizes these green dashed lines across all data points.

### **2.3. Gradient Descent Unlocked - Build Your First AI Model From Scratch**

step 1 : Open [Google Colab](https://colab.research.google.com/) account. 
step 2 : Expand + code. (Note: Add + code and click run button for adding code and executing code when ever necessary)
step 3 : Upload dataset file ( studentscores-1fef94ba-27e1-4fab-a7ad-56867b8fb5a1.csv ) provided.<br>

<img width="1920" height="1020" alt="Module 2-3 1" src="https://github.com/user-attachments/assets/3460fb83-eea5-4e8c-a0ec-04614bd7aa05" /><br>

step 4 : Import pandas, numpy and matplotlib libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

step 5 : Import pandas dataset

dataset = pd.read_csv('studentscores-1fef94ba-27e1-4fab-a7ad-56867b8fb5a1.csv')

step 6 : check the displayed values of dataset (rows and columns) to verify

print(dataset)<br>

<img width="1920" height="1020" alt="Module 2-3 2" src="https://github.com/user-attachments/assets/34858fbb-14c2-4cda-be36-cbdccf2831b1" /><br>

step 7 : check the scatter plot

dataset = pd.read_csv('studentscores-1fef94ba-27e1-4fab-a7ad-56867b8fb5a1.csv')
plt.scatter (dataset['Hours'], dataset['Scores'])
plt.show()<br>

<img width="543" height="413" alt="dataset scatter" src="https://github.com/user-attachments/assets/ae17cd14-87e4-4e1c-b2bb-80ab0e68e984" /><br>

step 7 : Separate the data set into X and Y coloumns for pre processing the data and verify the data points

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values
print(X)
print(Y)<br>

<img width="1920" height="1020" alt="Module 2-3 3" src="https://github.com/user-attachments/assets/ece135ed-d9a4-49c2-a6c6-95c9050b225b" /><br>

step 8 : Create a basic structure for a model

class Model():
      def __init__(self, learning_rate, iterations):
        # Instance attributes
        self.learning_rate = learning_rate
        self.iterations = iterations

The model class provides a basic structure for a model, initializing it with two key parameters: a learning_rate and the number of iterations. These parameters will be passed when creating an object of the model class. These typically represent hyperparameters used in machine learning models, such as the step size for optimization algorithms and the number of training cycles.These attributes are stored within each instance of the model class, allowing them to be used in subsequent methods for training, prediction, or other operations.

step 9 : Generate predictions

def predict(self, X):
            return X.dot(self.slope) + self.constant

It performs the core calculation of linear regression by multiplying the input feature matrix X by the learned slope (or weights) and adding the constant (or bias). This equation represents y = Xw + b, where y is the predicted output, X is the input features, w is the slope/weights, and b is the constant/bias

step 10 : Set up the model's parameters and train the model

def fit(self, X, Y):
            self.m, self.n = X.shape
            self.slope = np.zeros(self.n)
            self.constant = 0
            self.X = X
            self.Y = Y

            for i in range(self.iterations):
                self.update_weights()
            return self

step 11 : Update the model's parameters (slope/weights and constant/bias) based on the calculated gradients and a learning rate.

def update_weights(self):
            Y_pred = self.predict(self.X)
            dw = -(2*(self.X.T).dot(self.Y-Y_pred)) / self.m
            db = -2*np.sum(self.Y-Y_pred) / self.m

            self.slope = self.slope - self.learning_rate * dw
            self.constant = self.constant - self.learning_rate * db
            return self

step 12 : instantiate and train the machine learning model

model =Model(learning_rate=0.01, iterations=1000)
model.fit(X,Y)

step 12 : Predict the machine learning model.

Y_pred = model.predict(X)
print(Y_pred)

output: [26.91171724 52.33687281 33.75695143 85.58515317 36.69062323 17.13281125
 92.43038736 56.24843521 83.62937197 28.86749844 77.76202838 60.1599976
 46.46952922 34.73484203 13.22124886 89.49671557 26.91171724 21.04437365
 62.1157788  74.82835658 28.86749844 49.40320102 39.62429503 69.93890359
 78.73991898]

step 12 : Generate a plot that combines a scatter plot and a line plot.

plt.scatter (dataset['Hours'], dataset['Scores'])
plt.scatter(X, Y_pred, color="red")
plt.show()<br>
<img width="543" height="413" alt="scattered and line" src="https://github.com/user-attachments/assets/4c4d3b00-6f6c-4710-b6a3-770efc637ca4" /><br>


### **2.4. Visualizing Gradient Descent in Action**

step 1 : Generate slope and constant to be used in actual code to be uploaded into VSD Squadron board

print(model.slope, model.constant)

output : [9.77890599] 2.4644522714760995
