from bindings import *
import random
import math
import rod.conversions as conversions
import networkx as nx
import sklearn.neighbors
import numpy as np
import time

from rod.solvers.collision_detection import Collision_detector

# The number of nearest neighbors the endpoint will try to connect to
K = 10

# the radius by which the rod will be expanded
epsilon = FT(0.1)


# Calculate the scene's bounding box
def calc_bbox(obstacles):
    X = []
    Y = []
    for poly in obstacles:
        for point in poly.vertices():
            X.append(point.x())
            Y.append(point.y())
    min_x = min(X)
    max_x = max(X)
    min_y = min(Y)
    max_y = max(Y)

    return min_x, max_x, min_y, max_y


# Convert CGALPY's Point_d object into an array of doubles
def point_d_to_arr(p: Point_d):
    return [p[i].to_double() for i in range(p.dimension())]


# Given two angles theta_1, theta_2 in range [0, 2pi) and a direction (clockwise, or anti-clockwise),
# compute the distance between theta_1 and theta_2 in that direction. Result must be in range [0, 2pi)
def path_angular_dist(theta_1, theta_2, is_clockwise):
    # calculate clockwise distnaces:
    if theta_2 - theta_1 < -math.pi:
        angular_dist = theta_1 - theta_2
    elif theta_2 - theta_1 <= 0:
        angular_dist = theta_1 - theta_2
    elif theta_2 - theta_1 < math.pi:
        angular_dist = 2 * math.pi - theta_2 + theta_1
    else:
        angular_dist = 2 * math.pi - theta_2 + theta_1

    if not is_clockwise:
        if angular_dist != 0:
            angular_dist = 2 * math.pi - angular_dist

    return abs(angular_dist)


# distance used for nearest neighbor search
def custom_dist(p, q):
    rod_translational_dist = math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
    rod_rotational_dist = min(path_angular_dist(p[2], q[2], True), path_angular_dist(p[2], q[2], False))
    w_t, w_r = 1, 1
    weighted_dist = w_t * rod_translational_dist + w_r * rod_rotational_dist
    return weighted_dist


def find_nearest(p, points):
    p = point_d_to_arr(p)

    curr_nearest_point = points[0]
    curr_nearest_dist = custom_dist(p, point_d_to_arr(curr_nearest_point))
    for point in points:
        curr_point = point_d_to_arr(point)
        curr_dist = custom_dist(p, curr_point)
        if curr_dist < curr_nearest_dist:
            curr_nearest_dist = curr_dist
            curr_nearest_point = point

    return curr_nearest_point


def calc_direction_between_points(origin, target):
    origin_arr = point_d_to_arr(origin)
    target_arr = point_d_to_arr(target)

    direction = [0, 0, 0]
    for i in range(len(direction)):
        direction[i] = target_arr[i] - origin_arr[i]
    return direction


def calc_point_in_direction_from_point(origin, direction, step_size):
    origin_arr = point_d_to_arr(origin)
    target = [0, 0, 0]
    for i in range(len(target)):
        target[i] = origin_arr[i] + step_size * direction[i]

    return target

def clip_point_to_fit_in_range(p, x_range, y_range, z_range):
    p[0] = max(min(p[0], x_range[1]), x_range[0])
    p[1] = max(min(p[1], y_range[1]), y_range[0])
    p[2] = max(min(p[2], z_range[1]), z_range[0])
    return p




def generate_path(length, obstacles, origin, destination, argument, writer, isRunning):
    t0 = time.perf_counter()
    # Parsing of arguments
    path = []
    try:
        num_landmarks = int(argument)
    except Exception as e:
        print("argument is not an integer", file=writer)
        return path

    polygons = [conversions.tuples_list_to_polygon_2(p) for p in obstacles]
    bbox = calc_bbox(polygons)
    x_range = (bbox[0].to_double(), bbox[1].to_double())
    y_range = (bbox[2].to_double(), bbox[3].to_double())
    z_range = (0, 2 * math.pi)

    begin = Point_d(3, [FT(origin[0]), FT(origin[1]), FT(origin[2])])
    end = Point_d(3, [FT(destination[0]), FT(destination[1]), FT(destination[2])])

    # Initiate the graph
    G = nx.DiGraph()
    G.add_nodes_from([begin, end])
    points = [begin]  # originally: [begin, end]

    # Initiate the collision detector
    cd = Collision_detector(polygons, [], epsilon)

    # distance used for nearest neighbor search
    def edge_weight(p, q):
        base_rod_position = [p[0], p[1], p[0] + length.to_double() * math.cos(p[2]),
                             p[1] + length.to_double() * math.sin(p[2])]
        target_rod_position = [q[0], q[1], q[0] + length.to_double() * math.cos(q[2]),
                               q[1] + length.to_double() * math.sin(q[2])]

        sd = math.sqrt((base_rod_position[0] - target_rod_position[0]) ** 2 +
                       (base_rod_position[1] - target_rod_position[1]) ** 2 +
                       (base_rod_position[2] - target_rod_position[2]) ** 2 +
                       (base_rod_position[3] - target_rod_position[3]) ** 2)

        return sd

    # Sample landmarks
    done_flag = False
    full_eta_counter = 0
    i = 0
    prev_i = 0
    eta = 0.1
    eta_histogram = [0 for i in range(11)]

    while i < num_landmarks and not done_flag:
        rand_x, rand_y, rand_z = FT(random.uniform(x_range[0], x_range[1])),\
                                 FT(random.uniform(y_range[0], y_range[1])),\
                                 FT(random.uniform(z_range[0], z_range[1]))
        p_rand = Point_d(3, [rand_x, rand_y, rand_z])

        # find nearest point to p_rand of all sampled points
        p_near = find_nearest(p_rand, points)

        # steer towards p_rand from p_near with a step size of eta
        direction_towards_p_rand = calc_direction_between_points(p_near, p_rand)
        j = 10
        eta_successful = False
        while j > 0 and not eta_successful:
            p_new = calc_point_in_direction_from_point(p_near, direction_towards_p_rand, eta * j)
            p_new = clip_point_to_fit_in_range(p_new, x_range, y_range, z_range)
            if cd.is_rod_position_valid(FT(p_new[0]), FT(p_new[1]), FT(p_new[2]), length):
                eta_histogram[j] += 1
                p_new = Point_d(3, [FT(p_new[0]), FT(p_new[1]), FT(p_new[2])])
                for clockwise in (True, False):
                    if cd.is_rod_motion_valid(p_near, p_new, clockwise, length):
                        eta_successful = True
                        G.add_node(p_new)
                        points.append(p_new)
                        weight = edge_weight(point_d_to_arr(p_near), point_d_to_arr(p_new))
                        G.add_edge(p_near, p_new, weight=weight, clockwise=clockwise)
                        i += 1
            j -= 1

        if i % 1000 == 0 and prev_i != i:
            print("Running:", i, "landmarks sampled", file=writer)
            print("Eta histogram: ", eta_histogram, file=writer)
            prev_i = i

            _points = np.array([point_d_to_arr(p) for p in points])
            _K = K
            nearest_neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=_K, metric=custom_dist, algorithm='auto')
            nearest_neighbors.fit(_points)
            _, nn_indices = nearest_neighbors.kneighbors([point_d_to_arr(end)])
            for index in nn_indices[0]:
                for clockwise in (True, False):
                    potential_point = points[index]
                    if cd.is_rod_motion_valid(potential_point, end, clockwise, length):
                        G.add_node(end)
                        weight = edge_weight(point_d_to_arr(potential_point), point_d_to_arr(end))
                        G.add_edge(potential_point, end, weight=weight, clockwise=clockwise)
                        done_flag = True
                        break
        if i % 100 == 0 and prev_i != i:
            print("Running:", i, "landmarks sampled", file=writer)
            prev_i = i

    print(i, "landmarks sampled", file=writer)
    print(done_flag)

    if nx.has_path(G, begin, end):
        shortest_path = nx.shortest_path(G, begin, end)
        print("path found", file=writer)
        print("distance:", nx.shortest_path_length(G, begin, end, weight='weight'), file=writer)

        if len(shortest_path) == 0:
            return path
        first = shortest_path[0]
        path.append((first[0], first[1], first[2], True))
        for i in range(1, len(shortest_path)):
            last = shortest_path[i - 1]
            next = shortest_path[i]
            # determine correct direction
            clockwise = G.get_edge_data(last, next)["clockwise"]
            path.append((next[0], next[1], next[2], clockwise))
    else:
        print("no path was found", file=writer)
    t1 = time.perf_counter()
    print("Time taken:", t1 - t0, "seconds", file=writer)
    return path
