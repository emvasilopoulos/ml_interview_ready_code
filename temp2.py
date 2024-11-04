def calculate_dimensions_of_new_image(current_image_size, expected_image_size):
    x_w = current_image_size.width
    x_h = current_image_size.height
    expected_image_width = expected_image_size.width
    expected_image_height = expected_image_size.height
    if x_w <= expected_image_width and x_h <= expected_image_height:
        width_ratio = x_w / expected_image_width  # less than 1
        height_ratio = x_h / expected_image_height  # less than 1
        if width_ratio > height_ratio:
            new_height = int(x_h / width_ratio)
            resize_height = new_height
            resize_width = expected_image_width
            pad_dimension = 1
            expected_dimension_size = expected_image_height
        else:
            new_width = int(x_w / height_ratio)
            resize_height = expected_image_height
            resize_width = new_width
            pad_dimension = 2
            expected_dimension_size = expected_image_width
    elif x_w <= expected_image_width and x_h > expected_image_height:
        keep_ratio = x_w / x_h
        new_height = expected_image_height
        new_width = int(new_height * keep_ratio)
        resize_height = expected_image_height
        resize_width = new_width
        pad_dimension = 2
        expected_dimension_size = expected_image_width
    elif x_w > expected_image_width and x_h <= expected_image_height:
        keep_ratio = x_w / x_h
        new_width = expected_image_width
        new_height = int(new_width / keep_ratio)
        resize_height = new_height
        resize_width = expected_image_width
        pad_dimension = 1
        expected_dimension_size = expected_image_height
    else:
        width_ratio = x_w / expected_image_width  # greater than 1
        height_ratio = x_h / expected_image_height  # greater than 1
        if width_ratio > height_ratio:
            new_height = int(x_h / width_ratio)
            resize_height = new_height
            resize_width = expected_image_width
            pad_dimension = 1
            expected_dimension_size = expected_image_height
        else:
            new_width = int(x_w / height_ratio)
            resize_height = expected_image_height
            resize_width = new_width
            pad_dimension = 2

    return resize_width, resize_height, pad_dimension, expected_dimension_size
