"""Joint MRI modality and field labels shared by data, models and scripts."""

MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
NUM_CLASSES = len(MODALITIES) * len(FIELDS)


def get_class_label(modality, field):
    """Return the shared modality × field label in the range 0–14."""
    return MODALITIES.index(modality) * len(FIELDS) + FIELDS.index(field)
