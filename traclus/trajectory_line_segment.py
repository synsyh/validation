from traclus.distance_trajectory import all_distance


class ClusterCandidate:
    def __init__(self):
        self.cluster = None
        self.__is_noise = False

    def is_classified(self):
        return self.__is_noise or self.cluster != None

    def is_noise(self):
        return self.__is_noise

    def set_as_noise(self):
        self.__is_noise = True

    def assign_to_cluster(self, cluster):
        self.cluster = cluster
        self.__is_noise = False

    def distance_to_candidate(self, other_candidate):
        raise NotImplementedError()


class TrajectoryLineSegment(ClusterCandidate):
    def __init__(self, line_segment, position_in_trajectory=None,
                 id=None):
        ClusterCandidate.__init__(self)

        self.line_segment = line_segment
        self.position_in_trajectory = position_in_trajectory
        self.num_neighbors = -1
        self.id = id

    def get_num_neighbors(self):
        if self.num_neighbors == -1:
            raise Exception("haven't counted num neighbors yet")
        return self.num_neighbors

    def set_num_neighbors(self, num_neighbors):
        if self.num_neighbors != -1 and self.num_neighbors != num_neighbors:
            raise Exception("neighbors count should never be changing")
        self.num_neighbors = num_neighbors

    def distance_to_candidate(self, other_candidate):
        if other_candidate or other_candidate.line_segment or self.line_segment:
            raise Exception()
        return all_distance(self.line_segment, other_candidate.line_segment)


class TrajectoryLineSegmentFactory():
    def __init__(self):
        self.next_traj_line_seg_id = 0

    def new_trajectory_line_seg(self, line_segment):
        next_id = self.next_traj_line_seg_id
        self.next_traj_line_seg_id += 1
        return TrajectoryLineSegment(line_segment=line_segment, id=next_id)
