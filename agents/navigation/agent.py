#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a base class for autonomous agents in CARLA.
"""

from enum import Enum
import carla


class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of an agent.
    """

    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3
    BLOCKED_AT_STOP = 4


class Agent(object):
    """
    Base class for an agent that navigates the CARLA world.
    """

    def __init__(self, vehicle):
        """
        Initializes the Agent object.
        :param vehicle: The carla.Vehicle object to be controlled.
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._state = AgentState.NAVIGATING
        self._local_planner = None
        self._grp = None

    def get_world(self):
        """
        Returns the CARLA world object.
        """
        return self._world

    def set_destination(self, start_location, end_location=None):
        """
        Sets the target destination for the agent.
        Subclasses should implement this method to set up their routing logic.
        :param start_location: The starting carla.Location (or end_location if end_location is None).
        :param end_location: The target carla.Location (optional, for compatibility).
        """
        raise NotImplementedError

    def run_step(self, debug=False):
        """
        Executes one step of navigation.
        Subclasses should implement this method to return a carla.VehicleControl.
        :param debug: A boolean flag for debugging purposes.
        :return: A carla.VehicleControl object.
        """
        raise NotImplementedError

    def done(self):
        """
        Returns True if the agent has reached its destination.
        Subclasses should implement this method.
        :return: A boolean indicating if the destination has been reached.
        """
        raise NotImplementedError

    def emergency_stop(self):
        """
        Generates a carla.VehicleControl object for an emergency stop.
        :return: A carla.VehicleControl object with brake=1.0.
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        control.manual_gear_shift = False
        return control
