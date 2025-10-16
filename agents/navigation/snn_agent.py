#!/usr/bin/env python

# Copyright (c) 2024 World Explorer AI
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements a self-driving agent that uses a Spiking Neural Network
for depth estimation to perform basic collision avoidance.
"""

import numpy as np
import carla

from agents.navigation.behavior_agent import BehaviorAgent, AgentState


class SnnDepthAgent(BehaviorAgent):
    """
    SnnDepthAgent implements a self-driving agent that uses a SNN depth model
    to navigate and avoid obstacles.
    """

    def __init__(self, vehicle, model, device, behavior="cautious"):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param model: The pre-trained SNN depth estimation model.
            :param device: The torch device (cpu or cuda).
            :param behavior: type of agent to apply
        """
        super(SnnDepthAgent, self).__init__(vehicle, behavior)
        self.model = model
        self.device = device
        self._predicted_depth = None

        # --- Behavior Tuning ---
        # Override some of the default 'cautious' behavior parameters for better reactivity.
        if behavior == "cautious":
            self.behavior["braking_distance"] = 10  # Increase braking distance
            self.behavior["safety_time"] = 4  # Increase safety time

        # --- SNN Hazard Detection Parameters ---
        # Minimum distance to consider an object a hazard.
        self.hazard_distance_threshold = 8.0  # meters
        # Percentage of pixels in the ROI that must be below the threshold to trigger a stop.
        self.hazard_pixel_percentage_threshold = 0.05  # Be more sensitive

    def set_depth_prediction(self, depth_map):
        """
        Receives and stores the latest depth prediction from the main script.
        :param depth_map: A numpy array representing the predicted depth.
        """
        self._predicted_depth = depth_map

    def _is_hazard_ahead(self):
        """
        Analyzes the predicted depth map to detect immediate obstacles.

        :return: True if a hazard is detected, False otherwise.
        """
        if self._predicted_depth is None:
            return False

        depth_map = self._predicted_depth
        h, w = depth_map.shape

        # Define a Region of Interest (ROI) in the center of the view
        # This ROI is focused on the direct path ahead of the car.
        # It covers the middle 30% of the horizontal view.
        # Vertically, it starts from the middle of the screen (50%) and goes
        # down to 70%, avoiding the sky and the vehicle's hood.
        roi_x_start = int(w * 0.35)
        roi_x_end = int(w * 0.65)
        roi_y_start = int(h * 0.50)
        roi_y_end = int(h * 0.70)

        roi = depth_map[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        if roi.size == 0:
            return False

        # Find pixels indicating objects closer than the hazard distance threshold
        close_pixels = roi[roi < self.hazard_distance_threshold]

        # Calculate what percentage of the ROI is considered a hazard
        percentage_close = len(close_pixels) / roi.size

        if percentage_close > self.hazard_pixel_percentage_threshold:
            print(
                f"HAZARD DETECTED: Obstacle within {self.hazard_distance_threshold}m."
            )
            return True

        return False

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

        This method first checks for immediate hazards using the SNN depth map.
        If a hazard is detected, it triggers an emergency stop.
        Otherwise, it falls back to the parent BehaviorAgent's navigation logic,
        which handles route following, traffic lights, and other vehicles.

        :return: carla.VehicleControl
        """
        # 1. Highest priority: Check for immediate hazards using our SNN depth model
        if self._is_hazard_ahead():
            self._state = AgentState.BLOCKED_BY_VEHICLE
            return self.emergency_stop()

        # 2. If no immediate hazard, use the parent's more complex logic for navigation.
        control = super(SnnDepthAgent, self).run_step(debug=debug)

        return control
