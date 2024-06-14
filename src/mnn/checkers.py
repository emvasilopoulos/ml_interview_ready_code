image_formats = ["jpg", "png"]


def is_image_file(image_file: str):
    return any([image_file.endswith(format) for format in image_formats])


def is_labels_file(labels_file: str, format_suffix: str = "txt"):
    return labels_file.endswith(format_suffix)
