"""
CARLA Diverse Dataset Collection Script

This script collects a large-scale diverse dataset (25k+ frames) with:
- Multiple weather conditions (sunny day, night, rain, fog, sunset)
- Various camera configurations (FOV, position, resolution)
- Different maps and spawn points
- Autopilot with diverse driving scenarios

Usage:
    python scripts/collect_diverse_dataset.py --frames 25000 --output data/diverse_25k
"""

import argparse
import os
import sys
import time
import json
import queue
from pathlib import Path
from datetime import datetime
import random

import numpy as np
import cv2

try:
    import carla
except ImportError:
    print("ERROR: CARLA Python API not found!")
    print("Please install the CARLA wheel file:")
    print(
        "  uv pip install ~/carla_workspace/PythonAPI/carla/dist/carla-0.9.16-cp312-*.whl"
    )
    sys.exit(1)


# Weather presets for diversity
WEATHER_PRESETS = {
    "clear_day": {
        "cloudiness": 0.0,
        "precipitation": 0.0,
        "sun_altitude_angle": 45.0,
        "fog_density": 0.0,
        "wetness": 0.0,
    },
    "clear_noon": {
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "sun_altitude_angle": 90.0,
        "fog_density": 0.0,
        "wetness": 0.0,
    },
    "cloudy_day": {
        "cloudiness": 80.0,
        "precipitation": 0.0,
        "sun_altitude_angle": 45.0,
        "fog_density": 0.0,
        "wetness": 0.0,
    },
    "sunset": {
        "cloudiness": 30.0,
        "precipitation": 0.0,
        "sun_altitude_angle": 10.0,
        "fog_density": 0.0,
        "wetness": 0.0,
    },
    "night_clear": {
        "cloudiness": 0.0,
        "precipitation": 0.0,
        "sun_altitude_angle": -90.0,
        "fog_density": 0.0,
        "wetness": 0.0,
    },
    "night_cloudy": {
        "cloudiness": 70.0,
        "precipitation": 0.0,
        "sun_altitude_angle": -90.0,
        "fog_density": 0.0,
        "wetness": 0.0,
    },
    "rain_day": {
        "cloudiness": 90.0,
        "precipitation": 80.0,
        "sun_altitude_angle": 45.0,
        "fog_density": 10.0,
        "wetness": 90.0,
    },
    "rain_night": {
        "cloudiness": 90.0,
        "precipitation": 60.0,
        "sun_altitude_angle": -90.0,
        "fog_density": 10.0,
        "wetness": 80.0,
    },
    "fog_day": {
        "cloudiness": 50.0,
        "precipitation": 0.0,
        "sun_altitude_angle": 30.0,
        "fog_density": 70.0,
        "wetness": 0.0,
    },
    "fog_night": {
        "cloudiness": 50.0,
        "precipitation": 0.0,
        "sun_altitude_angle": -90.0,
        "fog_density": 50.0,
        "wetness": 0.0,
    },
}

# Camera configurations for diversity
CAMERA_CONFIGS = {
    "standard": {
        "fov": 90,
        "position": carla.Location(x=1.5, z=2.4),
        "rotation": carla.Rotation(pitch=0),
    },
    "wide_angle": {
        "fov": 110,
        "position": carla.Location(x=1.5, z=2.4),
        "rotation": carla.Rotation(pitch=0),
    },
    "narrow_angle": {
        "fov": 70,
        "position": carla.Location(x=1.5, z=2.4),
        "rotation": carla.Rotation(pitch=0),
    },
    "hood_mount": {
        "fov": 90,
        "position": carla.Location(x=2.0, z=1.0),
        "rotation": carla.Rotation(pitch=0),
    },
    "dashcam": {
        "fov": 90,
        "position": carla.Location(x=0.5, z=1.5),
        "rotation": carla.Rotation(pitch=0),
    },
    "slight_down": {
        "fov": 90,
        "position": carla.Location(x=1.5, z=2.4),
        "rotation": carla.Rotation(pitch=-5),
    },
    "slight_up": {
        "fov": 90,
        "position": carla.Location(x=1.5, z=2.4),
        "rotation": carla.Rotation(pitch=5),
    },
}


class DiverseDataCollector:
    """
    Collects diverse CARLA dataset with multiple weather and camera configurations.
    """

    def __init__(self, host="localhost", port=2000, resolution=(640, 480)):
        self.host = host
        self.port = port
        self.width, self.height = resolution

        # Data storage (cleared after each batch is saved)
        self.rgb_images = []
        self.depth_images = []
        self.metadata_list = []

        # CARLA objects
        self.client = None
        self.world = None
        self.vehicle = None
        self.rgb_camera = None
        self.depth_camera = None

        # Current configuration
        self.current_weather = None
        self.current_camera_config = None
        self.current_map = None

        # Synchronization - using Queues for robustness
        self.rgb_queue = queue.Queue()
        self.depth_queue = queue.Queue()
        self.total_collected = 0

    def connect(self):
        """Connect to CARLA simulator."""
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)

        self.world = self.client.get_world()
        self.current_map = self.world.get_map().name
        print(f"Connected to CARLA world: {self.current_map}")

        # Enable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        print("Synchronous mode enabled (20 FPS)")

    def set_weather(self, preset_name):
        """Set weather conditions."""
        if preset_name not in WEATHER_PRESETS:
            print(f"Warning: Unknown weather preset '{preset_name}'")
            return

        preset = WEATHER_PRESETS[preset_name]
        weather = carla.WeatherParameters(
            cloudiness=preset["cloudiness"],
            precipitation=preset["precipitation"],
            sun_altitude_angle=preset["sun_altitude_angle"],
            fog_density=preset["fog_density"],
            wetness=preset["wetness"],
        )
        self.world.set_weather(weather)
        self.current_weather = preset_name
        print(f"  Weather set to: {preset_name}")

    def cleanup_actors(self):
        """Destroy existing cameras and vehicle."""
        actors_to_destroy = []
        if self.rgb_camera:
            actors_to_destroy.append(self.rgb_camera)
            self.rgb_camera = None
        if self.depth_camera:
            actors_to_destroy.append(self.depth_camera)
            self.depth_camera = None
        if self.vehicle:
            actors_to_destroy.append(self.vehicle)
            self.vehicle = None

        if actors_to_destroy:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in actors_to_destroy]
            )

    def spawn_vehicle(self, vehicle_type="vehicle.tesla.model3"):
        """Spawn a vehicle at a random spawn point."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_type)[0]

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            print("ERROR: No spawn points available!")
            return False

        spawn_point = random.choice(spawn_points)

        # Try to spawn, retry if location is blocked
        for attempt in range(10):
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is not None:
                self.vehicle.set_autopilot(True)
                print(f"  Spawned vehicle: {vehicle_type}")
                return True
            else:
                print(
                    f"Warning: Spawn point blocked, retrying... (attempt {attempt + 1})"
                )
                spawn_point = random.choice(spawn_points)

        print("ERROR: Failed to spawn vehicle after 10 attempts")
        return False

    def setup_cameras(self, camera_config_name="standard"):
        """Setup cameras with specified configuration."""
        if camera_config_name not in CAMERA_CONFIGS:
            print(f"Warning: Unknown camera config '{camera_config_name}'")
            camera_config_name = "standard"

        config = CAMERA_CONFIGS[camera_config_name]
        blueprint_library = self.world.get_blueprint_library()

        # RGB Camera
        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(self.width))
        rgb_bp.set_attribute("image_size_y", str(self.height))
        rgb_bp.set_attribute("fov", str(config["fov"]))

        # Depth Camera
        depth_bp = blueprint_library.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", str(self.width))
        depth_bp.set_attribute("image_size_y", str(self.height))
        depth_bp.set_attribute("fov", str(config["fov"]))

        # Camera transform
        camera_transform = carla.Transform(config["position"], config["rotation"])

        # Spawn cameras
        self.rgb_camera = self.world.spawn_actor(
            rgb_bp, camera_transform, attach_to=self.vehicle
        )
        self.depth_camera = self.world.spawn_actor(
            depth_bp, camera_transform, attach_to=self.vehicle
        )

        # Register callbacks to queues
        self.rgb_camera.listen(self.rgb_queue.put)
        self.depth_camera.listen(self.depth_queue.put)

        self.current_camera_config = camera_config_name
        print(f"  Camera config: {camera_config_name} (FOV: {config['fov']})")

    def collect_frame(self):
        """Collect a single synchronized frame using queues."""
        self.world.tick()  # Advances the simulation one step

        try:
            # Retrieve data from queues. This will block until data is available.
            rgb_image = self.rgb_queue.get(timeout=2.0)
            depth_image = self.depth_queue.get(timeout=2.0)
        except queue.Empty:
            print("Warning: A sensor queue was empty. Skipping frame.")
            return False

        # --- Process RGB Image ---
        rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8).reshape(
            (self.height, self.width, 4)
        )
        rgb_final = cv2.cvtColor(rgb_array, cv2.COLOR_BGRA2RGB)

        # --- Process Depth Image ---
        depth_array = np.frombuffer(depth_image.raw_data, dtype=np.uint8).reshape(
            (self.height, self.width, 4)
        )
        # Apply the Logarithmic depth conversion formula from CARLA docs
        R = depth_array[:, :, 2].astype(np.float32)
        G = depth_array[:, :, 1].astype(np.float32)
        B = depth_array[:, :, 0].astype(np.float32)
        normalized_depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0**3 - 1.0)
        depth_final = normalized_depth * 1000.0  # Scale to meters

        # --- Store data ---
        self.rgb_images.append(rgb_final)
        self.depth_images.append(depth_final)

        # Get vehicle state for metadata
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        transform = self.vehicle.get_transform()

        # Store metadata
        metadata = {
            "frame_id": self.total_collected,
            "weather": self.current_weather,
            "camera_config": self.current_camera_config,
            "map": self.current_map,
            "speed": float(speed),
            "location": {
                "x": transform.location.x,
                "y": transform.location.y,
                "z": transform.location.z,
            },
            "rotation": {
                "pitch": transform.rotation.pitch,
                "yaw": transform.rotation.yaw,
                "roll": transform.rotation.roll,
            },
        }
        self.metadata_list.append(metadata)

        self.total_collected += 1
        return True

    def collect_diverse_dataset(
        self, total_frames, output_dir, batch_size=500, save_freq=5
    ):
        """
        Collect diverse dataset with varying conditions.

        Args:
            total_frames: Total number of frames to collect
            output_dir: Output directory
            batch_size: Frames per simulation run before changing conditions
            save_freq: How many simulation batches to run before saving data to disk
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"DIVERSE DATASET COLLECTION: {total_frames} frames")
        print(f"{'=' * 60}\n")

        weather_names = list(WEATHER_PRESETS.keys())
        camera_names = list(CAMERA_CONFIGS.keys())

        num_sessions = (total_frames + batch_size - 1) // batch_size
        print(f"Collection plan: {num_sessions} sessions of ~{batch_size} frames\n")

        start_time = time.time()

        for session_idx in range(num_sessions):
            remaining_frames = total_frames - self.total_collected
            frames_this_session = min(batch_size, remaining_frames)
            if frames_this_session <= 0:
                break

            print(
                f"\n--- Session {session_idx + 1}/{num_sessions} ({frames_this_session} frames) ---"
            )

            # Select random weather and camera config for diversity
            weather = random.choice(weather_names)
            camera_config = random.choice(camera_names)

            print(f"Configuration: Weather: {weather}, Camera: {camera_config}")

            self.cleanup_actors()
            self.set_weather(weather)
            if not self.spawn_vehicle():
                print("Failed to spawn vehicle, skipping session")
                continue
            self.setup_cameras(camera_config)

            # Wait for vehicle to stabilize
            for _ in range(20):
                self.world.tick()
                time.sleep(0.05)

            # Collect frames for this session
            print(f"Collecting frames: ", end="", flush=True)
            for i in range(frames_this_session):
                if not self.collect_frame():
                    print(f"\nWarning: Failed to collect frame {self.total_collected}")
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}", end="...", flush=True)
            print(f" ✓ Done")

            # Save data periodically to avoid high memory usage
            if (session_idx + 1) % save_freq == 0 or (session_idx + 1) == num_sessions:
                self._save_batch_data(output_path, session_idx + 1)

        elapsed = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"COLLECTION COMPLETE!")
        print(f"{'=' * 60}")
        print(f"Total frames collected: {self.total_collected}")
        print(f"Time elapsed: {elapsed / 60:.1f} minutes")
        print(f"Average FPS: {self.total_collected / elapsed:.1f}")

        # Final save of metadata and visualizations
        self._save_final_metadata(output_path)
        self.cleanup_actors()

    def _save_batch_data(self, output_path, session_num):
        """Save collected data for the current batch and clear memory."""
        if not self.rgb_images:
            print("  No new images to save.")
            return

        batch_file = output_path / f"batch_{session_num:03d}.npz"

        rgb_array = np.array(self.rgb_images, dtype=np.uint8)
        depth_array = np.array(self.depth_images, dtype=np.float32)

        np.savez_compressed(batch_file, rgb=rgb_array, depth=depth_array)

        print(
            f"  ✓ Saved {len(self.rgb_images)} frames to {batch_file.name} "
            f"({self.total_collected} frames total)"
        )

        # *** CRITICAL: Clear the lists to free up memory ***
        self.rgb_images.clear()
        self.depth_images.clear()

    def _save_final_metadata(self, output_path):
        """Save final metadata, stats, and sample visualizations."""
        print(f"\nSaving final metadata and samples to {output_path}...")

        # Save metadata
        metadata = {
            "num_frames": self.total_collected,
            "resolution": [self.width, self.height],
            "maps": list(set(meta["map"] for meta in self.metadata_list)),
            "weather_presets": list(WEATHER_PRESETS.keys()),
            "camera_configs": list(CAMERA_CONFIGS.keys()),
            "timestamp": datetime.now().isoformat(),
            "frame_metadata": self.metadata_list,
        }
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_file}")

        # Save statistics
        self._save_statistics(output_path)
        self._save_sample_visualizations(output_path)

    def _save_statistics(self, output_path):
        """Compute and save dataset statistics from metadata."""
        stats = {}
        weather_counts = {}
        for meta in self.metadata_list:
            weather = meta["weather"]
            weather_counts[weather] = weather_counts.get(weather, 0) + 1
        stats["weather_distribution"] = weather_counts

        camera_counts = {}
        for meta in self.metadata_list:
            camera = meta["camera_config"]
            camera_counts[camera] = camera_counts.get(camera, 0) + 1
        stats["camera_distribution"] = camera_counts

        stats_file = output_path / "statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Saved statistics: {stats_file}")

    def _save_sample_visualizations(self, output_path):
        """Save sample visualizations from different conditions."""
        print(
            "This function should be adapted to load data from saved batches if needed."
        )
        print("For now, it will not produce visualizations to avoid re-loading data.")
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Collect diverse CARLA dataset with varying weather and camera configs"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=25000,
        help="Total number of frames to collect (default: 25000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/diverse_25k",
        help="Output directory (default: data/diverse_25k)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Frames per simulation session before changing conditions (default: 1000)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=5,
        help="Save data to disk every N sessions (default: 5)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="640x480",
        help="Camera resolution (default: 640x480)",
    )

    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split("x"))

    # Create collector
    collector = DiverseDataCollector(
        host=args.host, port=args.port, resolution=(width, height)
    )

    original_settings = None
    try:
        # Connect to CARLA
        collector.connect()
        original_settings = collector.world.get_settings()

        # Collect diverse dataset
        collector.collect_diverse_dataset(
            total_frames=args.frames,
            output_dir=args.output,
            batch_size=args.batch_size,
            save_freq=args.save_freq,
        )

        print(f"\n{'=' * 60}")
        print("SUCCESS! Dataset ready for training.")
        print(f"Data saved in batches at: {args.output}")
        print(f"{'=' * 60}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if collector and collector.world:
            print("Restoring original world settings and cleaning up actors...")
            if original_settings:
                collector.world.apply_settings(original_settings)
            collector.cleanup_actors()


if __name__ == "__main__":
    main()
