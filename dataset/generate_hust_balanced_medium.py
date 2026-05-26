import generate_hust_mild as mild


num_clients = mild.num_clients
raw_dir_path = mild.raw_dir_path
dir_path = "dataset/hust_balanced_medium/"

train_ratio = mild.train_ratio
balanced_medium_alpha = 0.4
condition_affinity = 0.30
min_labels_per_client = 2

window_size = mild.window_size
window_stride = mild.window_stride
source_chunk_length = mild.source_chunk_length
signal_channel = mild.signal_channel
target_conditions = mild.target_conditions
label_names = mild.label_names


def generate_dataset(seed):
    original_dir_path = mild.dir_path
    original_alpha = mild.mild_alpha
    original_affinity = mild.condition_affinity
    original_min_labels = mild.min_labels_per_client
    original_variant = mild.dataset_variant
    original_display_name = mild.dataset_display_name

    mild.dir_path = dir_path
    mild.mild_alpha = balanced_medium_alpha
    mild.condition_affinity = condition_affinity
    mild.min_labels_per_client = min_labels_per_client
    mild.dataset_variant = "hust_balanced_medium"
    mild.dataset_display_name = "HUST-Balanced-Medium"

    try:
        mild.generate_dataset(seed)
    finally:
        mild.dir_path = original_dir_path
        mild.mild_alpha = original_alpha
        mild.condition_affinity = original_affinity
        mild.min_labels_per_client = original_min_labels
        mild.dataset_variant = original_variant
        mild.dataset_display_name = original_display_name


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generate_dataset(seed)
