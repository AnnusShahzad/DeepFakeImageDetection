from mscocoai.constants import (
    HF_DATASET_ID, LABEL_A_NAMES, LABEL_B_NAMES, NUM_ATTR_CLASSES, SPLIT_SIZES,
)


def test_label_names():
    assert LABEL_B_NAMES == ["real", "sd21", "sdxl", "sd3", "dalle3", "mj6"]
    assert LABEL_A_NAMES == ["real", "ai-generated"]
    assert NUM_ATTR_CLASSES == 6


def test_dataset_id_and_split_sizes():
    assert HF_DATASET_ID == "Rajarshi-Roy-research/Defactify_Image_Dataset"
    assert SPLIT_SIZES == {"train": 42000, "validation": 9000, "test": 45000}
