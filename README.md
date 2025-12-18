# Simple-Linear-regression-Salary-experience-data


### **Project: Linear Regression from Scratch with Gradient Descent**

This project demonstrates the fundamental principles of Linear Regression by building a model from scratch using Gradient Descent. The goal is to predict salaries based on years of experience, using a simple `Salary_Data.csv` dataset.

**Key Features:**

- **Custom Linear Regression Implementation**: A `MyLinearRegression` class is implemented from the ground up, featuring:
  - Initialization of learning rate (`lr`), number of iterations (`n_iters`), weights, and bias.
  - A `compute_cost` method to calculate the Mean Squared Error (MSE).
  - A `fit` method that employs Gradient Descent to iteratively update weights and bias, minimizing the cost function.
  - A `predict` method to generate predictions based on the learned parameters.

- **Data Preparation**: The dataset is loaded using pandas, and split into training and testing sets to ensure robust model evaluation.

- **Model Training**: The custom Linear Regression model is trained on the `X_train` and `y_train` data, with the cost history tracked to monitor convergence.

- **Performance Evaluation**: The model's performance is assessed on both training and testing datasets using key regression metrics:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R-squared (R2) Score**

- **Visualizations**: 
  - A scatter plot illustrates the training and testing data points, along with the model's best-fit line, providing a clear visual representation of the regression.
  - A plot of the cost function history demonstrates the optimization process during Gradient Descent.

- **Example Predictions**: The model is used to make predictions for new data points, showcasing its practical application.

This project serves as a foundational exercise in understanding linear regression, the mechanics of gradient descent, and the basic workflow of building and evaluating a machine learning model without relying on high-level libraries like scikit-learn for the core algorithm.
