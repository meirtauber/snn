#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a helper class for the GlobalRoutePlanner.
"""

import carla


class GlobalRoutePlannerDAO(object):
    """
    This class is the data access object for the GlobalRoutePlanner.
    It is responsible for fetching all the necessary data from the world.
    """

    def __init__(self, wmap, sampling_resolution):
        """
        Constructor
        :param wmap: carla.world object
        :param sampling_resolution: resolution of the waypoint sampling
        """
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap
        self._topology = None

    def get_topology(self):
        """
        Accessor for topology. This function retrieves the topology from the server.
        """
        if self._topology is None:
            self._topology = self._wmap.get_topology()
        return self._topology

    def get_waypoint(self, location):
        """
        The method is used to get the waypoint closest to the given location.
        :param location: carla.Location object
        :return: carla.Waypoint object
        """
        return self._wmap.get_waypoint(location)

    def get_resolution(self):
        """
        Accessor for sampling resolution.
        """
        return self._sampling_resolution
