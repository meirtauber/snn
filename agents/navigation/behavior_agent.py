#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights."""

import random
import numpy as np
import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.misc import get_speed


class BehaviorAgent(Agent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, behavior="normal"):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super(BehaviorAgent, self).__init__(vehicle)
        self.vehicle = vehicle
        self.behavior = None
        self._local_planner = None
        self._grp = None
        self.look_ahead_steps = 5

        # Vehicle information
        self.speed = 0
        self.speed_limit = 0
        self.direction = None
        self.incoming_direction = None
        self.incoming_waypoint = None
        self.start_waypoint = None
        self.end_waypoint = None

        self._is_at_traffic_light = 0
        self._has_traffic_light = False
        self._traffic_light_state = "Red"

        self._is_on_left = True
        self._left_lane_change_allowed = False
        self._right_lane_change_allowed = False

        self._incoming_vehicle_location = None
        self._incoming_vehicle_velocity = None

        self.set_behavior(behavior)

    def update_information(self):
        """
        This method is invoked whenever the world ticks. It handles the data
        update from the server side.
        """
        self.speed = get_speed(self.vehicle)
        self.speed_limit = self.vehicle.get_speed_limit()
        self._local_planner.set_speed(self.speed_limit)
        self.direction = self._local_planner.target_road_option
        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW

        self.incoming_direction = self._local_planner.incoming_road_option
        if self.incoming_direction is None:
            self.incoming_direction = RoadOption.LANEFOLLOW

        # Check if the vehicle is on the left lane
        current_waypoint = self._map.get_waypoint(self.vehicle.get_location())
        left_waypoint = current_waypoint.get_left_lane()
        self._is_on_left = (
            left_waypoint and left_waypoint.lane_type == carla.LaneType.Driving
        )

        # Check for lane change permissions
        self._left_lane_change_allowed = (
            current_waypoint.lane_change & carla.LaneChange.Left
        )
        self._right_lane_change_allowed = (
            current_waypoint.lane_change & carla.LaneChange.Right
        )

    def set_behavior(self, behavior_type):
        """
        Sets the agent's behavior.
        :param behavior_type: String, one of 'cautious', 'normal', 'aggressive'.
        """
        if behavior_type == "cautious":
            self.behavior = {
                "max_speed": 40,
                "speed_lim_dist": 6,
                "speed_decrease": 12,
                "safety_time": 3,
                "min_proximity_threshold": 12,
                "braking_distance": 6,
                "tailgate_counter": 0,
            }
        elif behavior_type == "normal":
            self.behavior = {
                "max_speed": 50,
                "speed_lim_dist": 3,
                "speed_decrease": 10,
                "safety_time": 2,
                "min_proximity_threshold": 10,
                "braking_distance": 5,
                "tailgate_counter": 0,
            }
        elif behavior_type == "aggressive":
            self.behavior = {
                "max_speed": 70,
                "speed_lim_dist": 1,
                "speed_decrease": 8,
                "safety_time": 1.5,
                "min_proximity_threshold": 8,
                "braking_distance": 4,
                "tailgate_counter": -1,
            }
        self._local_planner = LocalPlanner(self, self.behavior)

    def set_destination(self, start_location, end_location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router.
        """
        self.start_waypoint = self._map.get_waypoint(start_location)
        self.end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self._trace_route(self.start_waypoint, self.end_waypoint)
        self._local_planner.set_global_plan(route_trace)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint.
        """
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._map, 2.0)
            self._grp = GlobalRoutePlanner(dao)
            self._grp.setup()

        return self._grp.trace_route(
            start_waypoint.transform.location, end_waypoint.transform.location
        )

    def traffic_light_manager(self, waypoint):
        """
        This method is in charge of behaviors for red lights.
        """
        self._has_traffic_light = self.vehicle.is_at_traffic_light()
        if self._has_traffic_light:
            traffic_light = self.vehicle.get_traffic_light()
            self._traffic_light_state = str(traffic_light.get_state())
            if self._traffic_light_state == "Red":
                return True
        return False

    def _overtake(self, location, vehicle):
        """
        This method is in charge of overtaking behavior.
        """
        left_turn = self._left_lane_change_allowed
        right_turn = self._right_lane_change_allowed

        left_wpt = self._map.get_waypoint(location).get_left_lane()
        right_wpt = self._map.get_waypoint(location).get_right_lane()

        if (
            self.direction == RoadOption.CHANGELANELEFT
            or self.direction == RoadOption.CHANGELANERIGHT
        ) and (self._is_on_left and not right_turn):
            return True

        if (
            left_turn
            and left_wpt.lane_type == carla.LaneType.Driving
            and self.behavior["tailgate_counter"] > 0
        ):
            self.behavior["tailgate_counter"] = 0
            self.set_destination(
                left_wpt.transform.location, self.end_waypoint.transform.location
            )
            return True

        return False

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible overtaking attempts.
        """
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        def get_route_dist(route, location):
            min_dist = float("inf")
            for w, _ in route:
                dist = location.distance(w.transform.location)
                if dist < min_dist:
                    min_dist = dist
            return min_dist

        vehicle_state = False
        emergency_vehicle_state = False
        affected_vehicle = None

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self.vehicle.id:
                continue

            target_loc = target_vehicle.get_location()
            if (
                get_route_dist(self._local_planner.get_plan(), target_loc)
                > self.behavior["min_proximity_threshold"]
            ):
                continue

            if self._overtake(waypoint.transform.location, target_vehicle):
                continue

            if self.is_safe_to_cross(target_vehicle):
                emergency_vehicle_state = True
                affected_vehicle = target_vehicle
                break

            if self.is_vehicle_hazard(target_vehicle):
                vehicle_state = True
                affected_vehicle = target_vehicle
                break

        if emergency_vehicle_state:
            return True, affected_vehicle
        else:
            return vehicle_state, affected_vehicle

    def is_safe_to_cross(self, vehicle):
        """
        Check if the agent is in a junction and needs to yield to another vehicle.
        """
        if self._incoming_direction == RoadOption.RIGHT:
            if (
                self._incoming_vehicle_location
                and self._incoming_vehicle_location.distance(
                    self.vehicle.get_location()
                )
                < 15.0
            ):
                return True
        return False

    def is_vehicle_hazard(self, vehicle):
        """
        Check if a vehicle is a hazard to the ego vehicle based on distance and relative position.

        :param vehicle: The vehicle to check
        :return: True if the vehicle is a hazard, False otherwise
        """
        vehicle_location = vehicle.get_location()
        vehicle_velocity = vehicle.get_velocity()

        # Get ego vehicle info
        ego_location = self.vehicle.get_location()
        ego_velocity = self.vehicle.get_velocity()
        ego_speed = get_speed(self.vehicle)

        # Calculate distance to target vehicle
        distance = ego_location.distance(vehicle_location)

        # If vehicle is too far, it's not a hazard
        if distance > self.behavior["min_proximity_threshold"]:
            return False

        # Check if vehicle is in front of us
        ego_transform = self.vehicle.get_transform()
        ego_forward_vector = ego_transform.get_forward_vector()

        # Vector from ego to target vehicle
        relative_vector = vehicle_location - ego_location

        # Dot product to check if vehicle is ahead
        dot_product = (
            ego_forward_vector.x * relative_vector.x
            + ego_forward_vector.y * relative_vector.y
        )

        # If vehicle is behind us, it's not a hazard
        if dot_product < 0:
            return False

        # Calculate time to collision based on relative velocity
        relative_speed = ego_speed - get_speed(vehicle)

        # If we're slower than the vehicle ahead, no collision risk
        if relative_speed <= 0:
            return False

        # Calculate braking distance needed
        # Using simplified physics: d = v^2 / (2 * deceleration)
        # Assuming ~8 m/s^2 deceleration (~0.8g)
        braking_distance = (ego_speed / 3.6) ** 2 / (2 * 8.0)

        # Add safety margin from behavior settings
        safety_distance = braking_distance + self.behavior["braking_distance"]

        # Vehicle is a hazard if it's within our safety distance
        return distance < safety_distance

    def done(self):
        """
        Returns True if the agent has reached its destination.
        :return: A boolean indicating if the destination has been reached.
        """
        if self._local_planner is None:
            return True
        return self._local_planner.done()

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        self.update_information()
        control = None

        if self.behavior["tailgate_counter"] > 0:
            self.behavior["tailgate_counter"] -= 1

        ego_vehicle_loc = self.vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red light behavior
        if self.traffic_light_manager(ego_vehicle_wp):
            return self.emergency_stop()

        # 2: Collision and car avoidance behaviors
        vehicle_hazard, vehicle = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        if vehicle_hazard:
            return self.emergency_stop()

        # 3: Tailgating behavior
        if (
            vehicle
            and self.direction != RoadOption.CHANGELANELEFT
            and self.direction != RoadOption.CHANGELANERIGHT
        ):
            self.behavior["tailgate_counter"] = 5
            return self._local_planner.run_step(debug=debug)

        # 4: Intersection behavior
        if self.incoming_direction in (RoadOption.LEFT, RoadOption.RIGHT):
            self._state = AgentState.BLOCKED_BY_VEHICLE
            return self.emergency_stop()

        # 5: Lane change behavior
        if self._is_on_left and self._right_lane_change_allowed:
            self.set_destination(
                ego_vehicle_wp.get_right_lane().transform.location,
                self.end_waypoint.transform.location,
            )

        # 6: Normal behavior
        control = self._local_planner.run_step(debug=debug)

        return control
