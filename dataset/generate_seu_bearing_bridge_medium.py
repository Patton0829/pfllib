import generate_seu_bearing_medium as base


def generate_dataset(seed):
    original_dir_path = base.dir_path
    original_alpha = base.medium_alpha
    original_affinity = base.condition_affinity
    original_min_labels = base.min_labels_per_client
    original_variant = base.dataset_variant
    original_display_name = base.dataset_display_name

    base.dir_path = "dataset/seu_bearing_bridge_medium/"
    base.medium_alpha = 0.22
    base.condition_affinity = 0.20
    base.min_labels_per_client = 2
    base.dataset_variant = "seu_bearing_bridge_medium"
    base.dataset_display_name = "SEU-Bearing-Bridge-Medium"

    try:
        base.generate_dataset(seed)
    finally:
        base.dir_path = original_dir_path
        base.medium_alpha = original_alpha
        base.condition_affinity = original_affinity
        base.min_labels_per_client = original_min_labels
        base.dataset_variant = original_variant
        base.dataset_display_name = original_display_name


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generate_dataset(seed)
