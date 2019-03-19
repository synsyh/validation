import math


def determine_longer_and_shorter_lines(line_a, line_b):
    if line_a.length < line_b.length:
        return line_b, line_a
    else:
        return line_a, line_b


def all_distance(line_a, line_b):
    return perpendicular_distance(line_a, line_b) + angular_distance(line_a, line_b) + parrallel_distance(line_a,
                                                                                                          line_b)


def perpendicular_distance(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)
    dist_a = shorter_line.start.distance_to_projection_on(longer_line)
    dist_b = shorter_line.end.distance_to_projection_on(longer_line)

    if dist_a == 0.0 and dist_b == 0.0:
        return 0.0

    return (dist_a * dist_a + dist_b * dist_b) / (dist_a + dist_b)


def __perpendicular_distance(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)
    dist_a = longer_line.line.project(shorter_line.start).distance_to(shorter_line.start)
    dist_b = longer_line.line.project(shorter_line.end).distance_to(shorter_line.end)

    if dist_a == 0.0 and dist_b == 0.0:
        return 0.0
    else:
        return (math.pow(dist_a, 2) + math.pow(dist_b, 2)) / (dist_a + dist_b)


def angular_distance(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)
    sine_coefficient = shorter_line.sine_of_angle_with(longer_line)
    return abs(sine_coefficient * shorter_line.length)


def parrallel_distance(line_a, line_b):
    longer_line, shorter_line = determine_longer_and_shorter_lines(line_a, line_b)
    return min([longer_line.dist_from_start_to_projection_of(shorter_line.start),
                longer_line.dist_from_start_to_projection_of(shorter_line.end),
                longer_line.dist_from_end_to_projection_of(shorter_line.start),
                longer_line.dist_from_end_to_projection_of(shorter_line.end)])
