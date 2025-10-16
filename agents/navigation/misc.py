#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains miscellaneous helper functions for the CARLA agents.
"""

import math
import carla


def get_speed(vehicle):
    """
    Compute the speed of a vehicle in Km/h.

    :param vehicle: The vehicle for which speed is calculated.
    :return: Speed as a float in Km/h.
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


def is_within_distance(target_transform, current_transform, max_distance):
    """
    Check if a target transform is within a certain distance from a current transform.

    :param target_transform: The target transform.
    :param current_transform: The current transform.
    :param max_distance: The maximum allowed distance.
    :return: True if the distance is less than max_distance, False otherwise.
    """
    dist = current_transform.location.distance(target_transform.location)
    return dist < max_distance


def draw_waypoints(world, waypoints, z=0.5, life_time=1.0):
    """
    Draw a list of waypoints in the CARLA simulation for debugging purposes.

    :param world: The CARLA world object.
    :param waypoints: A list of CARLA waypoint objects.
    :param z: The z-coordinate offset for drawing the waypoints.
    :param life_time: The duration in seconds for which the waypoints will be visible.
    """
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=life_time)


def get_trafficlight_trigger_location(traffic_light):
    """
    Calculates the global location of a traffic light's trigger volume.

    :param traffic_light: The carla.TrafficLight actor.
    :return: The carla.Location of the trigger volume.
    """

    def rotate_point(point, angle):
        """
        Rotates a 3D point around the z-axis.
        """
        x_ = (
            math.cos(math.radians(angle)) * point.x
            - math.sin(math.radians(angle)) * point.y
        )
        y_ = (
            math.sin(math.radians(angle)) * point.x
            + math.cos(math.radians(angle)) * point.y
        )
        return carla.Vector3D(x_, y_, point.z)

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)

    # Correctly orient the trigger volume extent
    point = rotate_point(
        carla.Vector3D(0, 0, traffic_light.trigger_volume.extent.z), base_rot
    )
    point_location = area_loc + carla.Location(x=point.x, y=point.y)

    return point_location
