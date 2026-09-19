# MindTrack presentation notes

1. **Problem and scope:** lifestyle patterns and four dataset stress labels;
   educational use, not diagnosis.
2. **Dataset:** 5000 input rows, 4993 usable
   profiles; show the quality audit and explain unverified provenance.
3. **Preprocessing:** deduplication, invalid values, split isolation, imputation,
   scaling, and one-hot encoding; explain exclusion of Mental_Health_Score.
4. **Clustering:** show figures/clustering.png; explain why the selected
   k=2 does not need to equal the number of classes.
5. **Classification:** KNN/LR/RF, majority baseline, three-fold CV, validation
   macro F1 selection, then held-out evaluation.
6. **Results:** show metrics/classification.csv and confusion matrices.
   Primary: Random Forest; measured test macro F1: 0.8697.
7. **Demo:** run a valid assessment, compare model predictions, then show the
   research tab. Distinguish probabilities, global importance, and comparisons.
8. **Limitations and next steps:** source/units confirmation, calibration,
   independent validation, and reproducibility using saved runtime versions.

Use actual notebook outputs for screenshots. State clearly if presenting a
synthetic test fixture rather than the supplied project dataset.
