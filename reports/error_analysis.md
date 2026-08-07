# Error Analysis

              precision    recall  f1-score   support

     healthy       0.99      0.99      0.99       225
early_blight       0.94      0.95      0.95       183
 late_blight       0.95      0.94      0.95       196
   leaf_mold       0.99      0.98      0.99       196

    accuracy                           0.97       800
   macro avg       0.97      0.97      0.97       800
weighted avg       0.97      0.97      0.97       800


## Observations

- Overall model accuracy is high.
- Most confusion occurs between Early Blight and Late Blight because both have visually similar symptoms.
- Healthy and Leaf Mold classes achieve very high precision and recall.
- More data and stronger augmentation can further improve performance.
