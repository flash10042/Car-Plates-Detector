def intersection_over_union(box_1, box_2):
    left_x = max(box_1[0], box_2[0])
    lower_y = max(box_1[1], box_2[1])
    right_x = min(box_1[2], box_2[2])
    upper_y = min(box_1[3], box_2[3])

    intersection = max(0, right_x-left_x) * max(0, upper_y-lower_y)

    area_1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area_2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

    iou = intersection / (area_1 + area_2 - intersection)

    return iou