"""Dependency-free helpers for deterministic checkpoint evaluation."""


def flatten_split_records(split_clusters):
    """Return every split record in a stable cluster/chain order."""
    records = []
    for cluster_id in sorted(split_clusters):
        cluster_records = split_clusters[cluster_id]
        records.extend(
            list(item)
            for item in sorted(cluster_records, key=lambda item: tuple(str(value) for value in item))
        )
    return records
