# Breast Mass Classification: Logistic Regression and Decision Thresholds

**Course:** MSBA 265  
**Project:** Classification with Logistic Regression

## 1. Understanding Logistic Regression

I started by looking at four cases to understand how logistic regression works.

| Case | β₀ | β₁ | x | Threshold | Score (z) | Probability (p) | Class |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.0 | 0.8 | 2.0 | 0.50 | 0.6 | 0.6457 | 1 |
| B | -1.0 | 0.8 | 2.0 | 0.70 | 0.6 | 0.6457 | 0 |
| C | -1.0 | -0.8 | 2.0 | 0.50 | -2.6 | 0.0691 | 0 |
| D | 1.0 | 0.0 | 2.0 | 0.50 | 1.0 | 0.7311 | 1 |

**Case A calculation:**

Score: `z = -1.0 + (0.8 × 2.0) = 0.6`

Probability: `p = 1 / (1 + exp(-0.6)) = 0.6457`

Since 0.6457 is above 0.50, the prediction is Class 1.

Cases A and B have the same probability, but the prediction changes because the threshold is different. In Case C, changing the coefficient to a negative number makes the probability decrease as x increases. In Case D, the coefficient is zero, so x has no effect on the probability.

I learned that coefficients affect the probability, while the threshold determines the final class. Here, I changed the coefficients manually. During training, the model learns them from the data.

## 2. Dataset and Data Preparation

I used the Wisconsin Diagnostic Breast Cancer dataset from scikit-learn. It contains 569 samples with measurements of cell nuclei from breast masses.

For this assignment, I only used mean radius and mean texture. I changed the target so that 1 means malignant and 0 means benign. There were 212 malignant and 357 benign samples, with no missing values in either feature.

I split the data into 341 training, 114 validation, and 114 test records. I used stratified sampling and checked that the groups did not overlap.

I also used StandardScaler, but fitted it only on the training data. This prevents information from the validation or test sets from getting into the training process.

## 3. NumPy Logistic Regression

I built my own logistic regression model using NumPy and full-batch gradient descent.

I started with all coefficients at zero, used a learning rate of 0.1, and trained for 20,000 epochs without regularization or class weights.

The model uses the sigmoid function to turn scores into probabilities. Binary cross-entropy measures how well those probabilities match the actual labels. Gradient descent adjusts the coefficients to lower that loss. Lowering cross-entropy is the same as increasing the likelihood of the observed labels.

The initial training loss was 0.693147, and the final loss was 0.246834. The loss graph dropped quickly at first and then became almost flat.

![Training Loss](reports/training_loss.png)

## 4. Model Comparison

I compared my NumPy model with scikit-learn Logistic Regression and a DummyClassifier baseline.

| Coefficient | NumPy | Scikit-learn |
|---|---:|---:|
| Intercept | -0.727172 | -0.727172 |
| Mean radius | 3.656794 | 3.656794 |
| Mean texture | 1.009872 | 1.009872 |

The coefficients matched to six decimal places. The mean difference between their validation probabilities was only 0.0000000001.

For example, a one-standard-deviation increase in mean radius was associated with a 3.656794 increase in the predicted log-odds of malignancy, while keeping mean texture the same. This is not a fixed increase in probability or proof of causation.

The baseline predicted every validation sample as benign. Its accuracy was 0.622807, but it missed all 43 malignant samples. This showed me why accuracy alone can be misleading.

## 5. Threshold Selection and Evaluation

I tested seven thresholds on the validation data and selected 0.60 because it had the highest F1 score of 0.831169.

On the test data, the default threshold of 0.50 actually performed better. Its F1 was 0.805195, compared with 0.767123 at 0.60.

| Test metric | Threshold 0.50 | Threshold 0.60 |
|---|---:|---:|
| TN | 68 | 69 |
| FP | 4 | 3 |
| FN | 11 | 14 |
| TP | 31 | 28 |
| Accuracy | 0.868421 | 0.850877 |
| Precision | 0.885714 | 0.903226 |
| Recall | 0.738095 | 0.666667 |
| F1 | 0.805195 | 0.767123 |
| Log loss | 0.255657 | 0.255657 |

At 0.60, false positives dropped from 4 to 3, but false negatives increased from 11 to 14. In other words, the model made fewer false alarms but missed more malignant samples.

In the confusion matrix at 0.60, the model correctly identified 69 benign and 28 malignant samples. It incorrectly labeled 3 benign samples as malignant and 14 malignant samples as benign.

I kept 0.60 because it was selected before checking the test results. Changing the threshold did not change the predicted probabilities or log loss.

The selected F1 threshold is useful for this exercise, but it does not mean the costs of false positives and false negatives are equal in real medical situations.

## 6. Limitations and Conclusion

This model is only for educational use, not medical diagnosis.

It uses just two features and a relatively small historical dataset. I also did not check calibration or test the model on data from another source.

False negatives are especially important because a malignant sample could be incorrectly classified as benign.

Overall, I learned how logistic regression calculates probabilities, how gradient descent learns coefficients, and why choosing a threshold involves trade-offs. I also learned that a threshold that works best on validation data may not give the best test result.

---

## Research Process

I first checked the dataset, selected the two required features, and confirmed that there were no missing values. After splitting the data, I checked that every record belonged to exactly one group.

I fitted StandardScaler on the training data only and checked that the standardized training features had means close to zero and standard deviations of one.

Next, I tested my sigmoid and binary cross-entropy functions using small examples. Both returned the expected values.

After training my NumPy model, I checked the loss graph and confirmed that the training loss decreased. I also compared my coefficients and validation probabilities with scikit-learn. The coefficients matched to six decimal places, and the probability difference was almost zero.

One important result was that the threshold selected on validation data did not improve the test F1 score. I kept the original threshold rather than changing it after seeing the test results.

## References

Scikit-learn developers. (n.d.). *Breast cancer Wisconsin (diagnostic) dataset*. Scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. In *IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology* (pp. 861–870). SPIE.
