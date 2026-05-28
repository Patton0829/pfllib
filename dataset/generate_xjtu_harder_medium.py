import generate_xjtu_medium as medium


num_clients = medium.num_clients
raw_dir_path = medium.raw_dir_path
dir_path = "dataset/xjtu_harder_medium/"

train_ratio = medium.train_ratio
harder_medium_alpha = 0.2
condition_affinity = 0.20
min_labels_per_client = 2

window_size = medium.window_size
window_stride = medium.window_stride
label_names = medium.label_names


def generate_dataset(seed):
    original_dir_path = medium.dir_path
    original_alpha = medium.medium_alpha
    original_affinity = medium.condition_affinity
    original_min_labels = medium.min_labels_per_client
    original_variant = medium.dataset_variant
    original_display_name = medium.dataset_display_name

    medium.dir_path = dir_path
    medium.medium_alpha = harder_medium_alpha
    medium.condition_affinity = condition_affinity
    medium.min_labels_per_client = min_labels_per_client
    medium.dataset_variant = "xjtu_harder_medium"
    medium.dataset_display_name = "XJTU-Harder-Medium"

    try:
        medium.generate_dataset(seed)
    finally:
        medium.dir_path = original_dir_path
        medium.medium_alpha = original_alpha
        medium.condition_affinity = original_affinity
        medium.min_labels_per_client = original_min_labels
        medium.dataset_variant = original_variant
        medium.dataset_display_name = original_display_name


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generate_dataset(seed)
