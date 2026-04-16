"""
Synthetic OOD dataset package.

This package exists for one narrow purpose:
validate whether the current `FeatureGradCELoss` can improve
out-of-distribution generalization when we know which features are causal
and which are spurious.

Dataset design
==============

The synthetic dataset has 6 features:

- `causal_main`
- `causal_aux`
- `spurious_main`
- `noise_1`
- `noise_2`
- `noise_3`

Label generation
----------------

The binary label `y` is sampled first.
Then each informative feature is generated from `y` plus Gaussian noise.

- `causal_main`:
  strongest stable signal, correlated with `y` in every environment.
- `causal_aux`:
  weaker stable signal, also correlated with `y` in every environment.
- `spurious_main`:
  highly predictive in the training environment, but its correlation with
  `y` is flipped in the OOD environment.
- `noise_*`:
  pure Gaussian noise with no label information.

Environment shift
-----------------

There are two environments:

- train/validation environment:
  `spurious_main` has strong positive correlation with the label.
- OOD test environment:
  `spurious_main` has strong negative correlation with the label.

This makes the task useful for debugging OOD behavior:

- a baseline model can achieve high ID accuracy by using the spurious
  feature too much;
- a model encouraged to rely more on causal features should retain better
  OOD accuracy;
- a model encouraged to rely more on the spurious feature should perform
  worse on OOD.

Why validation matches training
-------------------------------

Validation is intentionally drawn from the same environment as training.
This matches the project definition:

- `ID` = validation accuracy
- `OOD` = accuracy on shifted test environments

That separation matters because the regularizer is selected using ID
performance while we inspect whether OOD changes in the expected direction.

Preprocessing
-------------

All splits are standardized using the training-set mean and standard
deviation only. This avoids leaking OOD statistics into the train pipeline
and keeps the setup aligned with normal ML practice.

Integration note
----------------

The synthetic dataset no longer uses a separate training core.
It plugs into the shared `src/main.py` and `src/trainer.py`, so ACSIncome
and synthetic experiments now share the same optimization and SHAP logic.
"""
