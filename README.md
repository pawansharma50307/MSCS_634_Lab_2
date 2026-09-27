# MSCS 634 – Lab 2: Classification Using KNN and RNN Algorithms

**Name:** Pawan Sharma
**Course:** MSCS-634-M50 – Advanced Big Data and Data Mining

## Purpose

This lab is intended to compare two neighbor-based classification algorithms, KNN and RNN, on the dataset Wine from scikit-learn. I trained two models using various parameter values (k and radius), measured the test accuracy for each model, made plots showing how accuracy depends on parameters, and compared the effect of parameters on the result.

This dataset contains 178 observations, 13 features and 3 types of wine. The splitting was stratified with a ratio 80/20 (142 for training and 36 for testing).

## Results

| k  | KNN Accuracy |
|----|--------------|
| 1  | 0.7778 |
| 5  | 0.8056 |
| 11 | 0.8056 |
| 15 | 0.8056 |
| 21 | 0.8056 |

| Radius | RNN Accuracy | Avg. Neighbors per Test Point |
|--------|--------------|-------------------------------|
| 350 | 0.7222 | 80.5 |
| 400 | 0.6944 | 87.9 |
| 450 | 0.6944 | 94.8 |
| 500 | 0.6944 | 100.1 |
| 550 | 0.6667 | 106.0 |
| 600 | 0.6667 | 111.0 |

## Key Insights

- **KNN showed better results than RNN** at all configurations tested. The highest result achieved by KNN is about 80.6% (for k ≥ 5), whereas for RNN the maximum value is about 72.2% at radius 350.
- **The performance of KNN first improved, then stabilized.** The configuration k = 1 is the worst because the presence of only one neighbor makes the classifier susceptible to noise. For values of k equal to 5 and greater the accuracy remains unchanged.
- **The performance of RNN worsened with increasing radius.** Already at radius 350, each test sample has about 80 out of 142 training samples within its radius. With larger radius, even more samples from other classes become included, thus destroying the local structure of the dataset.
- **It is generally safer to use KNN** as the default option because the value of k is independent of the units used for feature scaling. **RNN is helpful** when the fixed distance has a specific meaning in the problem domain or when outlier detection (samples without neighbors within the radius) is required.

## Challenges and Decisions

- **Feature scaling.** The features are highly dissimilar in scale, and `proline` (from ~278 to 1680) contributes significantly more to the Euclidean distance than any other feature. I used raw features for the experiments because the provided radiuses (350-600) make sense on the unscaled data. After standardization, the radius of 350 would include all the training points.
- **Extra scaling check.** To find out to what extent this matters, I did an additional experiment running KNN using `StandardScaler`. Accuracy reached 97-100%, which indicates that the unscaled distance is the only reason for mediocre performance.
- **Outliers in RNN.** The parameter `outlier_label="most_frequent"` was chosen so that the model wouldn’t break on test points that have no neighbors within the radius. No such test point was encountered for any value of radius.
- **Stratified split.** The classes are mildly imbalanced (59/71/48), so I added the parameter `stratify=y` to have approximately the same ratio in the training and testing datasets.

## Files

- `Lab_2.ipynb` – Jupyter Notebook with code, outputs, plots and analysis
- `README.md` – This file

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook Lab_2.ipynb
```