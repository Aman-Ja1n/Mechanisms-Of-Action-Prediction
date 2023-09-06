# Mechanisms-Of-Action-Prediction
The algorithm that classifies drugs based on their biological activity

1. **Kaggle Dataset Link** https://www.kaggle.com/competitions/lish-moa/data
2. **Kaggle Dataset Download Via API** `kaggle competitions download -c lish-moa`

# Dataset Description
In this competition, you will be predicting multiple targets of the Mechanism of Action (MoA) response(s) of different samples (sig_id), given various inputs such as gene expression data and cell viability data.

# Two notes:
The training data has an additional (optional) set of MoA labels that are not included in the test data and not used for scoring.
The re-run dataset has approximately 4x the number of examples seen in the Public test.

# Files

1. `train_features.csv` - Features for the training set. Features g- signify gene expression data, and c- signify cell viability data. cp_type indicates samples treated with a compound (cp_vehicle) or with a control perturbation (ctrl_vehicle); control perturbations have no MoAs; cp_time and cp_dose indicate treatment duration (24, 48, 72 hours) and dose (high or low).
2. `train_drug.csv` - This file contains an anonymous drug_id for the training set only.
3. `train_targets_scored.csv` - The binary MoA targets that are scored.
4. `train_targets_nonscored.csv` - Additional (optional) binary MoA responses for the training data. These are not predicted nor scored.
5. `test_features.csv` - Features for the test data. You must predict the probability of each scored MoA for each row in the test data.
6. `sample_submission.csv` - A submission file in the correct format.
