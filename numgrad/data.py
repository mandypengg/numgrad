"""Dataset loading and batching.

MNIST is fetched with ``sklearn.datasets.fetch_openml``. That is the only use of
scikit-learn in this project — it is a download helper, nothing more. No
sklearn model, metric, or preprocessing utility belongs here.

Pixels arrive as uint8 and are cast to ``float64`` and scaled to [0, 1] on the
way in. Labels are one-hot encoded as ``float64``.

Planned: load_mnist, train_test_split, one_hot, batches.

Not implemented yet.
"""
