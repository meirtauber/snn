#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a global route planner implementation based on A* search.
"""

import heapq
import carla
from agents.navigation.local_planner import RoadOption


class GlobalRoutePlanner(object):
    """
    This class provides a very high-level route plan.
    It computes the shortest path between two points on a CARLA map.
    """

    def __init__(self, dao):
        """
        :param dao: GlobalRoutePlannerDAO object
        """
        self._dao = dao
        self._topology = None
        self._graph = None
        self._id_map = None
        self._road_id_to_edge = None

    def setup(self):
        """
        Performs graph setup and waypoint extraction.
        """
        self._topology = self._dao.get_topology()
        self._graph, self._id_map, self._road_id_to_edge = self._build_graph()

    def _build_graph(self):
        """
        This function builds a networkx graph representation of the CARLA map.
        The graph is composed of nodes and edges that represent the waypoints and the roads.
        """
        graph = {}
        id_map = {}
        road_id_to_edge = {}

        for waypoint in self._topology:
            # Adding waypoint nodes
            id_map[waypoint[0].id] = waypoint[0]
            id_map[waypoint[1].id] = waypoint[1]

            if waypoint[0].id not in graph:
                graph[waypoint[0].id] = {}
            if waypoint[1].id not in graph:
                graph[waypoint[1].id] = {}

            # Adding edges
            graph[waypoint[0].id][waypoint[1].id] = waypoint[
                1
            ].transform.location.distance(waypoint[0].transform.location)
            graph[waypoint[1].id][waypoint[0].id] = waypoint[
                1
            ].transform.location.distance(waypoint[0].transform.location)

            # Storing road information
            if waypoint[0].road_id not in road_id_to_edge:
                road_id_to_edge[waypoint[0].road_id] = []
            road_id_to_edge[waypoint[0].road_id].append(
                (waypoint[0].id, waypoint[1].id)
            )

        return graph, id_map, road_id_to_edge

    def _find_closest_node(self, point):
        """
        A fast way to find the closest node to a point.
        """
        # Best candidate and its distance so far
        best_candidate = -1
        min_distance = float("inf")

        # Iterating over all nodes
        for node_id in self._graph.keys():
            distance = self._id_map[node_id].transform.location.distance(point)
            if distance < min_distance:
                min_distance = distance
                best_candidate = node_id

        return best_candidate

    def trace_route(self, origin, destination):
        """
        This method returns a list of waypoints connecting origin and destination.
        :param origin: carla.Location object of the starting point
        :param destination: carla.Location object of the destination
        :return: list of (carla.Waypoint, RoadOption) from origin to destination
        """
        start_node = self._find_closest_node(origin)
        end_node = self._find_closest_node(destination)

        route_ids = self._path_search(start_node, end_node)
        if route_ids is None:
            return []

        # We have the route as a list of IDs, now we need to convert this to a list of waypoints
        route = []
        for i in range(len(route_ids) - 1):
            start_wp = self._id_map[route_ids[i]]
            end_wp = self._id_map[route_ids[i + 1]]
            route.extend(self._trace_road_segment(start_wp, end_wp))

        return route

    def _trace_road_segment(self, start_wp, end_wp):
        """
        Traces a route between two waypoints.
        """
        segment_route = []
        current_wp = start_wp
        segment_route.append((current_wp, RoadOption.LANEFOLLOW))

        # We assume that the waypoints are on the same road and lane
        while current_wp.id != end_wp.id:
            next_wps = current_wp.next(2.0)  # 2m sampling
            if not next_wps:
                break
            current_wp = next_wps[
                0
            ]  # Assume there's only one next waypoint in a segment
            segment_route.append((current_wp, RoadOption.LANEFOLLOW))

        return segment_route

    def _path_search(self, start_node, end_node):
        """
        A* search algorithm.
        """
        frontier = [(0, start_node)]
        came_from = {}
        cost_so_far = {start_node: 0}

        came_from[start_node] = None

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == end_node:
                break

            for neighbor in self._graph[current]:
                new_cost = cost_so_far[current] + self._graph[current][neighbor]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._distance_heuristic(
                        self._id_map[end_node].transform.location,
                        self._id_map[neighbor].transform.location,
                    )
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if end_node not in came_from:
            return None

        return self._reconstruct_path(came_from, start_node, end_node)

    def _reconstruct_path(self, came_from, start, end):
        """
        Reconstructs the path from the came_from dictionary.
        """
        path = []
        current = end
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def _distance_heuristic(self, p1, p2):
        """
        Euclidean distance heuristic.
        """
        return p1.distance(p2)
