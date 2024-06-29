def read_string_label_from_txt(label_path: str) -> str:
    with open(label_path, "r") as f:
        return f.readlines()[0].replace("\n", "")


def read_float_value_from_txt(label_path: str) -> str:
    with open(label_path, "r") as f:
        return float(f.readlines()[0].replace("\n", ""))
