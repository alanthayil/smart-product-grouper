"""Evaluate clustering and canonical labels; produce report."""


def evaluate(clusters, canonical_labels):
    """Compute metrics and build report."""
    cluster_sizes: dict[str, int] = {}
    for record in clusters:
        cluster_id = str(int(record["cluster_id"]))
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    num_records = len(clusters)
    num_clusters = len(cluster_sizes)
    avg_cluster_size = (num_records / num_clusters) if num_clusters else 0.0
    largest_cluster = max(cluster_sizes.values()) if cluster_sizes else 0

    labels = {
        str(int(cluster_id)): label
        for cluster_id, label in canonical_labels.items()
    }
    return {
        "num_records": num_records,
        "num_clusters": num_clusters,
        "cluster_sizes": cluster_sizes,
        "labels": labels,
        "cluster_stats": {
            "total_clusters": num_clusters,
            "avg_cluster_size": avg_cluster_size,
            "largest_cluster": largest_cluster,
        },
    }
