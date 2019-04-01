import math
from traclus.distance_trajectory import angular_distance, perpendicular_distance
from traclus.line_segment import LineSegment

DISTANCE_OFFSET = 0.0000000001


def partition_cost(trajectory_line_segs, low, high, model_cost_func, encoding_cost_func):
    if low >= high:
        raise IndexError
    partition_line = LineSegment(trajectory_line_segs[low].start,
                                 trajectory_line_segs[high].start)
    model_cost = model_cost_func(partition_line)
    encoding_cost = encoding_cost_func(trajectory_line_segs, low, high, partition_line)
    return model_cost + encoding_cost


def no_partition_cost(trajectory_line_segs, low, high):
    if low >= high:
        raise IndexError
    total = 0.0
    for line_seg in trajectory_line_segs[low:high]:
        total += math.log(line_seg.length, 2)
    return total


def model_cost(partition_line):
    return math.log(partition_line.length, 2)


def encoding_cost(trajectory_line_segs, low, high,
                  partition_line, angular_dist_func, perpendicular_dist_func):
    total_angular = 0.0
    total_perp = 0.0
    for line_seg in trajectory_line_segs[low:high]:
        total_angular += angular_dist_func(partition_line, line_seg)
        total_perp += perpendicular_dist_func(partition_line, line_seg)

    return math.log(total_angular + DISTANCE_OFFSET, 2) + \
           math.log(total_perp + DISTANCE_OFFSET, 2)


def call_partition_trajectory(trajectory_point_list):
    if len(trajectory_point_list) < 2:
        raise ValueError

    def encoding_cost_func(trajectory_line_segs, low, high, partition_line):
        return encoding_cost(trajectory_line_segs, low, high,
                             partition_line=partition_line,
                             angular_dist_func=angular_distance,
                             perpendicular_dist_func=perpendicular_distance)

    def partition_cost_func(trajectory_line_segs, low, high):
        return partition_cost(trajectory_line_segs, low, high,
                              model_cost_func=model_cost,
                              encoding_cost_func=encoding_cost_func)

    # trajectory_line_segs = map(lambda i: LineSegment(trajectory_point_list[i], trajectory_point_list[i + 1]),
    #                            range(0, len(trajectory_point_list) - 1))
    trajectory_line_segs = [LineSegment(trajectory_point_list[i], trajectory_point_list[i + 1]) for i in
                            range(0, len(trajectory_point_list) - 1)]

    return partition_trajectory(trajectory_line_segs=trajectory_line_segs,
                                partition_cost_func=partition_cost_func,
                                no_partition_cost_func=no_partition_cost)


def partition_trajectory(trajectory_line_segs,
                         partition_cost_func,
                         no_partition_cost_func):
    if len(list(trajectory_line_segs)) < 1:
        raise ValueError
    low = 0
    partition_points = [0]
    last_pt = trajectory_line_segs[len(list(trajectory_line_segs)) - 1].end
    trajectory_line_segs.append(LineSegment(last_pt, last_pt))

    for high in range(2, len(trajectory_line_segs)):
        if trajectory_line_segs[high - 2].unit_vector.almost_equals(trajectory_line_segs[high - 1].unit_vector):
            continue
        elif not trajectory_line_segs[high].start.almost_equals(trajectory_line_segs[low].start):
            partition_cost = partition_cost_func(trajectory_line_segs, low, high)
            no_partition_cost = no_partition_cost_func(trajectory_line_segs, low, high)
            if partition_cost > no_partition_cost:
                partition_points.append(high - 1)
                low = high - 1

    partition_points.append(len(trajectory_line_segs) - 1)
    return partition_points


def call_partition_trajectory(trajectory_point_list):
    if len(trajectory_point_list) < 2:
        raise ValueError

    def encoding_cost_func(trajectory_line_segs, low, high, partition_line):
        return encoding_cost(trajectory_line_segs, low, high,
                             partition_line=partition_line,
                             angular_dist_func=angular_distance,
                             perpendicular_dist_func=perpendicular_distance)

    def partition_cost_func(trajectory_line_segs, low, high):
        return partition_cost(trajectory_line_segs, low, high,
                              model_cost_func=model_cost,
                              encoding_cost_func=encoding_cost_func)

    # trajectory_line_segs = map(lambda i: LineSegment(trajectory_point_list[i], trajectory_point_list[i + 1]),
    #                            range(0, len(trajectory_point_list) - 1))
    trajectory_line_segs = [LineSegment(trajectory_point_list[i], trajectory_point_list[i + 1]) for i in
                            range(0, len(trajectory_point_list) - 1)]

    return partition_trajectory(trajectory_line_segs=trajectory_line_segs,
                                partition_cost_func=partition_cost_func,
                                no_partition_cost_func=no_partition_cost)
