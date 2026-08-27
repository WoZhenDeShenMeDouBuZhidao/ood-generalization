"""Generated OOD datasets with known stable-feature oracle importance.

The benchmark contains balanced `simple`, `range`, `categorical_integer`,
`categorical_onehot`, and `multi_spurious` variants. Each uses an ID test group
and four increasingly difficult spurious-correlation shifts. Categorical
encodings share latent samples, and one-hot oracle weights preserve semantic
group mass across expanded columns.
"""
