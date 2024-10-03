# Machine Learning for Deforestation Detection in Colombia

## Overview

This project applies machine learning techniques to detect deforestation in Colombia using satellite imagery. The project uses various machine learning models, including Support Vector Machines (SVM), Random Forest, and Convolutional Neural Networks (CNN), to classify satellite images into two categories: deforested areas and non-deforested areas. This work is part of an academic project for the course *Introduction to Machine Learning for Economists* at the National University of Colombia. (The training images are not included in this repository (in totality) due to their large size.) A detailed report of the project can be found in the `main.tex` file.

## Objective

The primary objective of this project is to explore how effectively computer vision models, trained on satellite images, can detect deforestation. The focus is on evaluating different machine learning models and understanding their performance in detecting deforested areas in Colombia.

## Methodology

The project follows these main steps:

1. **Data Collection**:  
   Satellite images were collected from the Global Forest Change (GFC) dataset developed by Hansen et al. (2013). The dataset was processed using Google Earth Engine to label regions as deforested or not deforested.

2. **Preprocessing**:  
   The satellite images were preprocessed for use in machine learning models. The images were converted to grayscale, normalized, and resized to 64x64 pixels for consistency. This step ensures the data is in the appropriate format for training the models.

3. **Models Used**:
   - **Support Vector Machines (SVM)**: Used to classify images into deforested and non-deforested areas using radial basis function kernels (RBF).
   - **Random Forest (RF)**: A tree-based ensemble method used to improve classification performance and reduce overfitting.
   - **Convolutional Neural Networks (CNN)**: These models were used for their ability to capture spatial patterns in images.
   - **Transfer Learning**: ResNet50, a pre-trained model, was fine-tuned to improve classification performance.

4. **Evaluation**:  
   The models were evaluated using metrics like accuracy, precision, recall, and F1-score. The results highlight the strengths and limitations of each model in detecting deforestation.

## Results

The Random Forest model trained on color images performed the best with an accuracy of **67.96%**, followed closely by the **Transfer Learning (ResNet50)** model with an accuracy of **66.85%**. SVM models and CNNs also showed promise but underperformed compared to Random Forest and Transfer Learning.

## Future Improvements

Some potential improvements for the future include:
- Expanding the dataset to include more images and higher resolution data.
- Addressing temporal inconsistencies, as some regions may have been deforested in previous years but were not included in the dataset.
- Using more advanced models or hyperparameter tuning techniques like Random Search or Bayesian optimization for more efficient exploration of hyperparameter space.

## Files

- `main.tex`: The LaTeX file containing the detailed report, methodology, results, and discussion.
- `code/`: Contains the scripts used for data preprocessing, model training, and evaluation.
- `images/`: Contains example satellite images used in the project, including deforested and non-deforested areas.
- `README.md`: This file.

## Requirements

To run the project, you will need the following packages installed:

- Python 3.8+
- NumPy
- scikit-learn
- TensorFlow (for CNN models)
- Earth Engine Python API (for data extraction)

To install the required Python libraries, run:

```bash
pip install -r requirements.txt
```

## Running the Project

1. **Preprocess the data**:  
   Run the preprocessing script to extract and prepare the satellite images.

2. **Train the models**:  
   Use the scripts in the `code/` folder to train the SVM, Random Forest, and CNN models on the preprocessed data.

3. **Evaluate the models**:  
   Evaluate the models using the test dataset and review the metrics to determine the best-performing model.

## Author

This project was developed by Thomas Prada as part of the course *Introduction to Machine Learning for Economists* at the National University of Colombia.

## References

- Hansen, M. C., et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. *Science, 342*(6160), 850-853. https://doi.org/10.1126/science.1244693
- Jaiswal, S. (2024). Normalization in machine learning: Techniques and importance. DataCamp. https://www.datacamp.com/tutorial/normalization-in-machine-learning
- Sanderson, G. (2022). Convolutions. 3Blue1Brown. https://www.3blue1brown.com/lessons/convolutions
- WWF. (2024). Causas y consecuencias de la deforestación en Colombia. World Wildlife Fund Colombia. https://www.wwf.org.co/?386550/deforestacion-colombia-causas-consecuencias

