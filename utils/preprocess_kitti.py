import os
import numpy as np
from tqdm import tqdm
import argparse
import glob


def read_calib_file(filepath):
    """Reads a calibration file and returns a dictionary of matrices."""
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            if (
                ":" not in line
            ):  # Skip lines that don't contain a colon (e.g., comments)
                continue
            key, value = line.split(":", 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass  # skip non-numeric lines
    return data


def get_calib_matrices(calib_dir):
    """
    Loads relevant calibration matrices for KITTI.
    Specifically for cam_02 (left color camera).
    """
    calib_cam_to_cam_path = os.path.join(calib_dir, "calib_cam_to_cam.txt")
    calib_velo_to_cam_path = os.path.join(calib_dir, "calib_velo_to_cam.txt")

    # Read cam_to_cam calibration
    cam_to_cam_data = read_calib_file(calib_cam_to_cam_path)

    # P_rect_02 (3x4) is the rectified projection matrix for camera 2
    # Projects from rectified camera 00 coordinates to rectified camera 02 image plane.
    P_rect_02 = cam_to_cam_data["P_rect_02"].reshape(3, 4)

    # R_rect_00 (3x3) is the rectification matrix for cam_00
    R_rect_00 = cam_to_cam_data["R_rect_00"].reshape(3, 3)

    # Read velo_to_cam calibration
    velo_to_cam_data = read_calib_file(calib_velo_to_cam_path)
    # R_velo_to_cam (3x3) and T_velo_to_cam (3x1)
    R_velo_to_cam = velo_to_cam_data["R"].reshape(3, 3)
    T_velo_to_cam = velo_to_cam_data["T"].reshape(3, 1)

    # Create the rigid body transformation matrix from velodyne to cam_00
    # Add a row [0,0,0,1] to make it a 4x4 matrix
    RT_velo_to_cam_00 = np.eye(4)
    RT_velo_to_cam_00[:3, :3] = R_velo_to_cam
    RT_velo_to_cam_00[:3, 3:] = T_velo_to_cam

    return P_rect_02, RT_velo_to_cam_00, R_rect_00


def load_velodyne_points(filepath):
    """Loads Velodyne points from a .bin file."""
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    # Only x, y, z are needed for projection, ignore intensity
    return points[:, :3]


def project_lidar_to_depth_map(
    P_rect_02, RT_velo_to_cam_00, R_rect_00, lidar_points, image_shape
):
    """
    Projects LiDAR points to a dense depth map.
    image_shape: (height, width) of the target image.
    """
    height, width = image_shape

    # 1. Transform Velodyne points to Camera 00 coordinate system
    # Add homogeneous coordinate (1) to LiDAR points
    lidar_points_hom = np.hstack(
        (lidar_points, np.ones((lidar_points.shape[0], 1)))
    )  # N x 4

    # velo -> cam_00 (unrectified)
    # Result is 3D points in camera 00 (unrectified) frame
    cam00_points = np.dot(RT_velo_to_cam_00, lidar_points_hom.T).T[:, :3]  # N x 3

    # 2. Rectify points (unrectified cam_00 -> rectified cam_00)
    # R_rect_00 is a 3x3 matrix for rectification
    rect_cam00_points = np.dot(R_rect_00, cam00_points.T).T  # N x 3

    # Filter out points behind the camera (depth should be positive)
    valid_indices = rect_cam00_points[:, 2] > 0
    rect_cam00_points = rect_cam00_points[valid_indices]

    if rect_cam00_points.shape[0] == 0:
        return np.zeros(image_shape, dtype=np.float32)

    # Depths are the Z values in the rectified camera 00 frame
    depths = rect_cam00_points[:, 2]

    # 3. Project rectified 3D points (from cam_00) to 2D image plane (rectified cam_02)
    # P_rect_02 is 3x4 and projects from rectified cam_00 coordinates to rectified cam_02 image.
    # We need to add a homogeneous coordinate to rect_cam00_points for P_rect_02 (3x4 matrix)
    rect_cam00_points_hom = np.hstack(
        (rect_cam00_points, np.ones((rect_cam00_points.shape[0], 1)))
    )  # N x 4

    points_2d_hom = np.dot(P_rect_02, rect_cam00_points_hom.T).T  # N x 3

    # Perform perspective division
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:]  # N x 2 (u, v)

    # Filter points outside image boundaries
    u = points_2d[:, 0]
    v = points_2d[:, 1]

    valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_mask].astype(int)
    v = v[valid_mask].astype(int)
    depths = depths[valid_mask]

    # Create depth map
    depth_map = np.zeros(image_shape, dtype=np.float32)

    # Fill depth map. For multiple points in the same pixel, take the minimum depth.
    # Sort points by depth in ascending order. This way, when we iterate and assign,
    # shallower points will overwrite deeper ones if they fall into the same pixel.
    sort_idx = np.argsort(depths)  # Sort ascending
    depth_map[v[sort_idx], u[sort_idx]] = depths[sort_idx]

    return depth_map


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess KITTI raw data to generate depth maps from LiDAR."
    )
    parser.add_argument(
        "--kitti_root_dir",
        type=str,
        required=True,
        help="Root directory of the downloaded KITTI raw dataset (e.g., /path/to/kitti_raw_data).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated depth maps (e.g., /path/to/kitti_processed_depth).",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=1242,
        help="Target image width for depth map (e.g., KITTI 1242).",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=375,
        help="Target image height for depth map (e.g., KITTI 375).",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all 'date_drive_sync' directories directly under kitti_root_dir
    # Find all 'date_drive_sync' directories, handling both direct and nested structures.
    # The structure can be kitti_root_dir/date_drive_sync/ or kitti_root_dir/date/date_drive_sync/.
    drive_dirs = sorted(
        glob.glob(os.path.join(args.kitti_root_dir, "**", "*_sync"), recursive=True)
    )
    if not drive_dirs:
        print(
            f"No drive directories found in {args.kitti_root_dir}. Please check the path and structure."
        )
        return

    for drive_dir in tqdm(drive_dirs, desc="Processing KITTI drives"):
        date_drive_name = os.path.basename(
            drive_dir
        )  # e.g., 2011_09_26_drive_0001_sync
        # Extract date from the drive name (e.g., "2011_09_26" from "2011_09_26_drive_0001_sync")
        date_name = date_drive_name.split("_drive")[0]

        # Calibration files are located in the date subdirectory under kitti_root_dir.
        # This assumes the date folder (e.g., '2011_09_26') is directly under kitti_root_dir.
        calib_dir = os.path.join(args.kitti_root_dir, date_name)
        image_dir = os.path.join(drive_dir, "image_02", "data")
        velodyne_dir = os.path.join(drive_dir, "velodyne_points", "data")

        output_drive_dir = os.path.join(
            args.output_dir, date_name, date_drive_name, "depth_maps"
        )
        os.makedirs(output_drive_dir, exist_ok=True)

        if not os.path.exists(image_dir) or not os.path.exists(velodyne_dir):
            print(
                f"Skipping {drive_dir}: Missing image_02/data or velodyne_points/data."
            )
            continue

        try:
            P_rect_02, RT_velo_to_cam_00, R_rect_00 = get_calib_matrices(calib_dir)
        except FileNotFoundError:
            print(f"Skipping {drive_dir}: Calibration files not found in {calib_dir}.")
            continue
        except Exception as e:
            print(f"Error loading calibration for {drive_dir}: {e}")
            continue

        # Get the list of Velodyne files
        velodyne_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))

        for i, velo_file in enumerate(
            tqdm(velodyne_files, desc=f"Drive {date_drive_name}", leave=False)
        ):
            image_shape = (args.image_height, args.image_width)

            lidar_points = load_velodyne_points(velo_file)
            depth_map = project_lidar_to_depth_map(
                P_rect_02, RT_velo_to_cam_00, R_rect_00, lidar_points, image_shape
            )

            # Save depth map as .npy file
            # The filename will correspond to the velodyne scan, e.g., 0000000000.npy
            output_filepath = os.path.join(
                output_drive_dir,
                f"{os.path.splitext(os.path.basename(velo_file))[0]}.npy",
            )
            np.save(output_filepath, depth_map)

    print("KITTI depth map preprocessing complete!")


if __name__ == "__main__":
    main()
