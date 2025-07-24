import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from jax import jit, vmap


import numpy as np
from typing import Optional, Union, List, Tuple
from pyproj import Transformer

jax.config.update("jax_enable_x64", True)

class SensorPosition():
    def __init__(self, dx: float, dy: float, dz: float):
        self.dx = dx
        self.dy = dy
        self.dz = dz

class SensorOrientation():
    def __init__(self, domega: float, dphi: float, dkappa: float):
        self.domega = domega
        self.dphi = dphi
        self.dkappa = dkappa

class ImageMeta():
    def __init__(self, width: int, height: int, pixsize: float, focal_length: float):
        self.width = width
        self.height = height
        self.pixsize = pixsize
        self.focal_length = focal_length

class Sensor():
    def __init__(self, id: int, position: SensorPosition, orientation: SensorOrientation, focal_length: float):
        self.id = id
        self.position = position
        self.orientation = orientation
        self.focal_length = focal_length

class GeoFrame():
    CAMERA_MODELS = {
        "lb4": {
            "sensors": [
                {
                    "pos": 0,
                    "dx": 0.0, "dy": 0.0, "dz": 0.0,
                    "dphi": 0.0, "dkappa": 0.0, "domega": 0.0,
                    "focal_length": 4.2589,
                    "mosaic": {"latmin": -50, "latmax": 45, "longmin": -36, "longmax": 36}
                },
                {
                    "pos": 1,
                    "dx": 0.059, "dy": 0.0, "dz": 0.042,
                    "dphi": -71.907528, "dkappa": -0.560417, "domega": -0.747527,
                    "focal_length": 4.2622,
                    "mosaic": {"latmin": -50, "latmax": 45, "longmin": 36, "longmax": 108}
                },
                {
                    "pos": 2,
                    "dx": 0.035, "dy": 0.0, "dz": 0.111,
                    "dphi": -144.178867, "dkappa": 0.273426, "domega": 0.27623,
                    "focal_length": 4.2598,
                    "mosaic": {"latmin": -50, "latmax": 45, "longmin": 108, "longmax": 180}
                },
                {
                    "pos": 3,
                    "dx": -0.037, "dy": 0.0, "dz": 0.11,
                    "dphi": -216.133942, "dkappa": 0.148435, "domega": 0.067258,
                    "focal_length": 4.2711,
                    "mosaic": {"latmin": -50, "latmax": 45, "longmin": -180, "longmax": -108}
                },
                {
                    "pos": 4,
                    "dx": -0.058, "dy": 0.0, "dz": 0.042,
                    "dphi": -287.957784, "dkappa": 0.021902, "domega": 0.095168,
                    "focal_length": 4.2693,
                    "mosaic": {"latmin": -50, "latmax": 45, "longmin": -108, "longmax": -36}
                },
                {
                    "pos": 5,
                    "dx": 0.0, "dy": 0.076, "dz": 0.061,
                    "dphi": -0.276425, "dkappa": 180.070803, "domega": 90.03968,
                    "focal_length": 4.2496,
                    "mosaic": {"latmin": 45, "latmax": 90, "longmin": -180, "longmax": 180}
                }
            ]
        },
        "lb7": {
            "sensors": [
                {
                    "pos": 0,
                    "dx": 0.0, "dy": 0.0, "dz": 0.0,
                    "dphi": 0.0, "dkappa": 0.0, "domega": 0.0,
                    "focal_length": 6.9523,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": -36, "longmax": 36}
                },
                {
                    "pos": 1,
                    "dx": 0.081, "dy": 0.0, "dz": 0.059,
                    "dphi": -71.408048, "dkappa": -0.187507, "domega": -0.18868,
                    "focal_length": 6.9605,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": 36, "longmax": 108}
                },
                {
                    "pos": 2,
                    "dx": 0.05, "dy": 0.0, "dz": 0.155,
                    "dphi": -35.877048, "dkappa": 179.885125, "domega": 179.966422,
                    "focal_length": 6.9843,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": 108, "longmax": 180}
                },
                {
                    "pos": 3,
                    "dx": -0.05, "dy": 0.0, "dz": 0.155,
                    "dphi": 35.686974, "dkappa": 179.9219, "domega": 179.989853,
                    "focal_length": 6.9582,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": -180, "longmax": -108}
                },
                {
                    "pos": 4,
                    "dx": -0.081, "dy": 0.0, "dz": 0.059,
                    "dphi": 71.811261, "dkappa": -0.301029, "domega": 0.107712,
                    "focal_length": 6.9669,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": -108, "longmax": -36}
                },
                {
                    "pos": 5,
                    "dx": 0.0, "dy": 0.098, "dz": 0.085,
                    "dphi": 0.019793, "dkappa": 179.895394, "domega": 90.036577,
                    "focal_length": 6.991,
                    "mosaic": {"latmin": 48.5, "latmax": 90, "longmin": -180, "longmax": 180}
                }
            ]
        },
        "lb8": {
            "sensors": [
                {
                    "pos": 0,
                    "dx": 0.0, "dy": 0.0, "dz": 0.0,
                    "dphi": 0.0, "dkappa": 0.0, "domega": 0.0,
                    "focal_length": 6.9475,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": -36, "longmax": 36}
                },
                {
                    "pos": 1,
                    "dx": 0.081, "dy": 0.0, "dz": 0.06,
                    "dphi": -71.925635, "dkappa": -0.066317, "domega": -0.003691,
                    "focal_length": 6.9396,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": 36, "longmax": 108}
                },
                {
                    "pos": 2,
                    "dx": 0.049, "dy": 0.001, "dz": 0.156,
                    "dphi": -143.834084, "dkappa": 0.272609, "domega": 0.418854,
                    "focal_length": 6.9629,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": 108, "longmax": 180}
                },
                {
                    "pos": 3,
                    "dx": -0.051, "dy": 0.001, "dz": 0.154,
                    "dphi": -215.966141, "dkappa": 0.139956, "domega": 0.010938,
                    "focal_length": 6.9424,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": -180, "longmax": -108}
                },
                {
                    "pos": 4,
                    "dx": -0.081, "dy": 0.0, "dz": 0.058,
                    "dphi": -287.842047, "dkappa": 0.260135, "domega": -0.126164,
                    "focal_length": 6.9581,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": -108, "longmax": -36}
                },
                {
                    "pos": 5,
                    "dx": -0.001, "dy": 0.095, "dz": 0.086,
                    "dphi": -0.15884, "dkappa": 179.999686, "domega": 90.133725,
                    "focal_length": 6.9459,
                    "mosaic": {"latmin": 48.5, "latmax": 90, "longmin": -180, "longmax": 180}
                }
            ]
        },
        "lb10": {
            "sensors": [
                {
                    "pos": 0,
                    "dx": 0.0, "dy": 0.0, "dz": 0.0,
                    "dphi": 0.0, "dkappa": 0.0, "domega": 0.0,
                    "focal_length": 6.9666,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": -36, "longmax": 36}
                },
                {
                    "pos": 1,
                    "dx": 0.081, "dy": 0.001, "dz": 0.059,
                    "dphi": -72.0007, "dkappa": 0.766589, "domega": 0.83249,
                    "focal_length": 6.9602,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": 36, "longmax": 108}
                },
                {
                    "pos": 2,
                    "dx": 0.05, "dy": 0.0, "dz": 0.155,
                    "dphi": -36.053938, "dkappa": 179.809987, "domega": -179.999046,
                    "focal_length": 6.9518,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": 108, "longmax": 180}
                },
                {
                    "pos": 3,
                    "dx": -0.05, "dy": 0.0, "dz": 0.156,
                    "dphi": 35.6003, "dkappa": 179.644969, "domega": -179.722306,
                    "focal_length": 6.9445,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": -180, "longmax": -108}
                },
                {
                    "pos": 4,
                    "dx": -0.081, "dy": 0.0, "dz": 0.059,
                    "dphi": 72.018611, "dkappa": 0.571209, "domega": -0.604061,
                    "focal_length": 6.9489,
                    "mosaic": {"latmin": -80, "latmax": 48.5, "longmin": -108, "longmax": -36}
                },
                {
                    "pos": 5,
                    "dx": 0.0, "dy": 0.098, "dz": 0.085,
                    "dphi": 0.131969, "dkappa": 179.782896, "domega": 89.923159,
                    "focal_length": 6.9521,
                    "mosaic": {"latmin": 48.5, "latmax": 90, "longmin": -180, "longmax": 180}
                }
            ]
        }
    }


    def __init__(self, 
                 easting: float,
                 northing: float,
                 height: float,
                 omega: float,
                 phi: float,
                 kappa: float,
                 imagemeta: ImageMeta,
                 camera_model: str):

        if camera_model not in self.CAMERA_MODELS:
            raise ValueError(f"Unknown camera_model '{camera_model}'. Must be one of {list(self.CAMERA_MODELS.keys())}")

        self.easting = easting
        self.northing = northing
        self.height = height
        self.omega = omega
        self.phi = phi
        self.kappa = kappa
        self.imagemeta = imagemeta

        # self.omega, self.phi, self.kappa = np.deg2rad(self.rot_from_wgs84())

        self.cubefaces = {
            0: {"domega": 0.0, "dphi": 0.0, "dkappa": 0.0},
            1: {"domega": 0.0, "dphi": 180.0, "dkappa": 0.0},
            2: {"domega": 0.0, "dphi": 90.0, "dkappa": 0.0},
            3: {"domega": 0.0, "dphi": 270.0, "dkappa": 0.0},
            4: {"domega": 90.0, "dphi": 0.0, "dkappa": 0.0},
            5: {"domega": 270.0, "dphi": 0.0, "dkappa": 0.0}
        }

        # Initialize sensors and multi_head_mask from camera_model metadata
        model_meta = self.CAMERA_MODELS[camera_model]
        self.sensors = []
        self.multi_head_mask = {}
        for sensor in model_meta["sensors"]:
            self.sensors.append(
                Sensor(
                    id=sensor["pos"],
                    position=SensorPosition(
                        dx=sensor["dx"],
                        dy=sensor["dy"],
                        dz=sensor["dz"]
                    ),
                    orientation=SensorOrientation(
                        domega=np.deg2rad(sensor["domega"]),
                        dphi=np.deg2rad(sensor["dphi"]),
                        dkappa=np.deg2rad(sensor["dkappa"])
                    ),
                    focal_length=sensor["focal_length"]
                )
            )
            self.multi_head_mask[sensor["pos"]] = sensor["mosaic"]

    @staticmethod
    def get_rotation_matrix(omega: float, phi: float, kappa: float) -> np.ndarray:
        mat_rot = np.zeros((4, 4))
        mat_rot[0, 0] = np.cos(phi) * np.cos(kappa)
        mat_rot[0, 1] = -np.cos(phi) * np.sin(kappa)
        mat_rot[0, 2] = np.sin(phi)
        mat_rot[1, 0] = np.cos(omega) * np.sin(kappa) + np.sin(omega) * np.sin(phi) * np.cos(kappa)
        mat_rot[1, 1] = np.cos(omega) * np.cos(kappa) - np.sin(omega) * np.sin(phi) * np.sin(kappa)
        mat_rot[1, 2] = -np.sin(omega) * np.cos(phi)
        mat_rot[2, 0] = np.sin(omega) * np.sin(kappa) - np.cos(omega) * np.sin(phi) * np.cos(kappa)
        mat_rot[2, 1] = np.sin(omega) * np.cos(kappa) + np.cos(omega) * np.sin(phi) * np.sin(kappa)
        mat_rot[2, 2] = np.cos(omega) * np.cos(phi)
        mat_rot[3, 3] = 1
        return mat_rot

    def get_LatLon(self, world: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Given world coordinates in the local CRS, calculate their longitude and latitude
        with respect to the frame's orientation (i.e., after applying the frame's rotation).
        Returns the longitude and latitude of the input coordinates in the frame's oriented CRS.

        Args:
            world: (N, 3) or (3,) array-like of world coordinates in the local CRS

        Returns:
            np.ndarray: (N, 2) array of [longitude, latitude] for each input point in the frame's orientation
        """
        # Ensure input is (N, 3)
        world = np.atleast_2d(world)
        if world.shape[1] != 3:
            raise ValueError("Input world coordinates must be of shape (3,) or (N, 3)")

        self.frame_x, self.frame_y, self.frame_z = self.world_to_frame(world, 0).T
        longitude = np.degrees(np.arctan2(self.frame_x, - self.frame_z))
        hyp = np.sqrt(self.frame_x**2 + self.frame_y**2)
        latitude = np.degrees(np.arctan2(self.frame_y, hyp))
        return np.column_stack([latitude, longitude])
    
    def latlon_to_sensor_id(self, latlon: Union[List[float], Tuple[float, float], np.ndarray]) -> np.ndarray:
        """
        Map latitude/longitude pairs to sensor IDs based on multi_head_mask.
        Accepts a single (lat, lon) or an array of shape (N, 2).
        Returns a single sensor ID or an array of sensor IDs.
        """

        latlon = np.asarray(latlon)

        sensor_ids = np.full(latlon.shape[0], -1, dtype=int)
        for key, current in self.multi_head_mask.items():
            shift = GeoFrame.mod(current["longmin"], 360)
            _longmax = GeoFrame.mod(current["longmax"] - shift, 360)
            lats = latlon[:, 0]
            lons = latlon[:, 1]
            _lons = GeoFrame.mod(lons - shift, 360)
            mask = (
                (_lons <= _longmax) &
                (lats >= current["latmin"]) &
                (lats <= current["latmax"]) &
                (sensor_ids == -1)
            )
            if np.any(mask):
                try:
                    sensor_ids[mask] = int(key)
                except Exception:
                    sensor_ids[mask] = key
        # Return scalar if input was 1D, else array
        return sensor_ids


    def get_face_id(self) -> Union[int, np.ndarray]:

        points = np.stack([self.frame_x, self.frame_y, self.frame_z], axis=1)

        # For each cubeface, compute the forward direction in world coordinates
        face_dirs = [
            [0.0, 0.0, -1.0],
            [0.0, 0.0,  1.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0,  0.0],
            [0.0, 1.0,  0.0],
            [0.0, -1.0, 0.0]
        ]
        face_dirs = np.array(face_dirs)  # (num_faces, 3)

        # For each point, find the face whose forward direction is closest (max dot product)
        dots = points @ face_dirs.T  # (N, num_faces)
        face_indices = np.argmax(dots, axis=1)

        # Return scalar if input was 1D, else array
        return int(face_indices[0]) if face_indices.size == 1 else face_indices
    
    def rot_from_wgs84(self):
        local_to_wgs84 = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
        wgs84_easting, wgs84_northing, wgs84_height = local_to_wgs84.transform(self.easting, self.northing, self.height)
        wgs84_to_local = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
        coords_plus_easting, coords_plus_northing, coords_plus_height = wgs84_to_local.transform(wgs84_easting, wgs84_northing + 1, wgs84_height)
        delta_x = coords_plus_easting - self.easting
        delta_y = coords_plus_northing - self.northing
        meridian_convergence = np.rad2deg(np.arctan2(delta_x, delta_y))
        return GeoFrame._rot_wgs84_metric(self.omega, self.phi, self.kappa, -meridian_convergence)

    def get_sensor_by_id(self, sensor_id: int) -> Optional[Sensor]:
        for sensor in self.sensors:
            if sensor.id == sensor_id:
                return sensor
        return None

    def get_world_ori_for_sensor_id(self, sensor_id: int) -> List[float]:
        sensor = self.get_sensor_by_id(sensor_id)
        if sensor is None:
            raise ValueError(f"Sensor with ID {sensor_id} not found")
        model = np.zeros(3)
        world = model.copy()
        R_c_c0 = self.get_rotation_matrix(
            sensor.orientation.domega,
            sensor.orientation.dphi,
            sensor.orientation.dkappa
        )
        r_c0_c = np.array([sensor.position.dx, sensor.position.dy, sensor.position.dz])
        world = GeoFrame.apply_matrix4(R_c_c0, world) + r_c0_c
        R_c0_w = self.get_rotation_matrix(
            self.omega,
            self.phi,
            self.kappa
        )
        r_w_c0 = np.array([self.easting, self.northing, self.height])
        world = GeoFrame.apply_matrix4(R_c0_w, world)
        R_c_w = R_c0_w @ R_c_c0
        world += r_w_c0
        rot = self.get_opk_from_rotation_matrix(R_c_w)
        return [world[0], world[1], world[2], rot[0], rot[1], rot[2]]

    def sensor_id_from_model(self, model: Union[np.ndarray, List[float]], face_id: int):

        frame = model.copy()
        R_c_c0 = self.get_rotation_matrix(
            np.deg2rad(self.cubefaces[face_id]["domega"]),
            np.deg2rad(self.cubefaces[face_id]["dphi"]),
            np.deg2rad(self.cubefaces[face_id]["dkappa"])
        )
        frame_x, frame_y, frame_z = GeoFrame.apply_matrix4(R_c_c0, frame).T
    
        longitude = np.degrees(np.arctan2(frame_x, - frame_z))
        hyp = np.sqrt(frame_x**2 + frame_y**2)
        latitude = np.degrees(np.arctan2(frame_y, hyp))
        sensor_ids = self.latlon_to_sensor_id(np.column_stack([latitude, longitude]))
        return sensor_ids

    def world_to_frame(self, world: Union[np.ndarray, List[float]], sensor_id: int) -> np.ndarray:
        sensor = self.get_sensor_by_id(sensor_id)
        if sensor is None:
            raise ValueError(f"Sensor with ID {sensor_id} not found")
        if isinstance(world, list):
            world = np.array(world)
        frame = world.copy()
        R_c0_w = self.get_rotation_matrix(
            self.omega,
            self.phi,
            self.kappa
        )
        r_w_c0 = np.array([self.easting, self.northing, self.height])
        frame -= r_w_c0
        frame = GeoFrame.apply_matrix4(R_c0_w.T, frame)
        # depth = np.sqrt(np.sum(frame**2, axis=1))
        return frame
    
    def frame_to_model(self, frame: np.ndarray, sensor_id: int, face_id: int) -> np.ndarray:
        sensor = self.get_sensor_by_id(sensor_id)
        if sensor is None:
            raise ValueError(f"Sensor with ID {sensor_id} not found")
        model = frame.copy()
        R_c_c0 = self.get_rotation_matrix(
            np.deg2rad(self.cubefaces[face_id]["domega"]),
            np.deg2rad(self.cubefaces[face_id]["dphi"]),
            np.deg2rad(self.cubefaces[face_id]["dkappa"])
        )
        r_c0_c = np.array([sensor.position.dx, sensor.position.dy, sensor.position.dz])
        model -= r_c0_c
        model = GeoFrame.apply_matrix4(R_c_c0.T, model)
        return model

    def model_to_image(self, model: np.ndarray) -> np.ndarray:
        model = model.reshape(-1, 3)
        image = np.zeros((model.shape[0], 2))
        image[:, 0] = self.imagemeta.focal_length * -1.0 * (model[:, 0] / model[:, 2])
        image[:, 1] = self.imagemeta.focal_length * -1.0 * (model[:, 1] / model[:, 2])
        return image

    def image_to_sensor(self, image: np.ndarray) -> np.ndarray:
        sensor_xy = np.zeros((image.shape[0], 2))
        sensor_xy[:, 0] = image[:, 0] / (self.imagemeta.pixsize) + self.imagemeta.width / 2.0
        sensor_xy[:, 1] = image[:, 1] / (-1.0 * self.imagemeta.pixsize) + self.imagemeta.height / 2.0
        return sensor_xy
    
    def sensor_to_image(self, sensor: np.ndarray) -> np.ndarray:
        image_xy = np.zeros((sensor.shape[0], 2))
        image_xy[:, 0] = (sensor[:, 0] - self.imagemeta.width / 2.0) * self.imagemeta.pixsize * 0.001
        image_xy[:, 1] = -1 * (sensor[:, 1] - self.imagemeta.height / 2.0) * self.imagemeta.pixsize * 0.001
        return image_xy


    def image_to_model(self, image: np.ndarray) -> np.ndarray:
        model = np.zeros((image.shape[0], 3))
        model[:, 0] = image[:, 0]
        model[:, 1] = image[:, 1]
        model[:, 2] = -1 * self.imagemeta.focal_length
        length = np.sqrt(np.sum(model**2, axis=1))
        model /= length

        return model

    def model_to_frame(self, model: np.ndarray, face_id: int) -> np.ndarray:
        sensor_ids = self.sensor_id_from_model(model, face_id)
        unique_sensor_ids = np.unique(sensor_ids)
        frames = np.zeros_like(model)
        for sensor_id in unique_sensor_ids:
            indices = np.where(sensor_ids == sensor_id)[0]
            sensor = self.get_sensor_by_id(sensor_id)
            if sensor is None:
                raise ValueError(f"Sensor with ID {sensor_id} not found")
            R_c_c0 = self.get_rotation_matrix(
                np.deg2rad(self.cubefaces[face_id]["domega"]),
                np.deg2rad(self.cubefaces[face_id]["dphi"]),
                np.deg2rad(self.cubefaces[face_id]["dkappa"])
            )
            r_c0_c = np.array([sensor.position.dx, sensor.position.dy, sensor.position.dz])
            frame_subset = model[indices].copy()
            frame_subset = GeoFrame.apply_matrix4(R_c_c0, frame_subset)
            frame_subset += r_c0_c
            frames[indices] = frame_subset
        return frames

    def frame_to_world(self, frame: np.ndarray) -> np.ndarray:
        world = frame.copy()
        R_c0_w = self.get_rotation_matrix(
            self.omega,
            self.phi,
            self.kappa
        )
        r_w_c0 = np.array([self.easting, self.northing, self.height])
        world = GeoFrame.apply_matrix4(R_c0_w, world)
        world += r_w_c0
        return world


    def get_opk_from_rotation_matrix(self, R: np.ndarray) -> List[float]:
        Ra = R[:3, :3]
        r13 = Ra[0, 2]
        r23 = Ra[1, 2]
        r33 = Ra[2, 2]
        r11 = Ra[0, 0]
        r12 = Ra[0, 1]
        phi = np.arctan2(r13, np.sqrt(r23**2 + r33**2))
        cos_phi = np.cos(phi)
        if cos_phi != 0.0:
            omega = np.arctan2(-r23 / cos_phi, r33 / cos_phi)
            kappa = np.arctan2(-r12 / cos_phi, r11 / cos_phi)
        else:
            r22 = Ra[1, 1]
            r32 = Ra[2, 1]
            if phi == np.pi / 2:
                omega = 0.0
                kappa = np.arctan2(r22, r32)
            else:
                omega = 0.0
                kappa = -np.arctan2(r22, r32)
        return [omega, phi, kappa] 

    @staticmethod
    def mod(n, m):
        """
        Computes the mathematical modulus of n modulo m, ensuring the result is always in [0, m).
        This handles negative n correctly, unlike Python's default % operator for negative numbers.

        Args:
            n: The dividend (can be negative or positive).
            m: The modulus (must be positive).

        Returns:
            The result of n mod m, always in the range [0, m).
        """
        return ((n % m) + m) % m

    @staticmethod
    def _opk_to_matrix(o, p, k):
        so = np.sin(o)
        sp = np.sin(p)
        sk = np.sin(k)
        co = np.cos(o)
        cp = np.cos(p)
        ck = np.cos(k)
        mat = np.array([
            [cp * ck, -cp * sk, sp],
            [co * sk + so * sp * ck, co * ck - so * sp * sk, -so * cp],
            [so * sk - co * sp * ck, so * ck + co * sp * sk, co * cp]
        ])
        return mat

    @staticmethod
    def _matrix_to_opk(matrix):
        p = np.arctan2(matrix[2,0], np.sqrt(matrix[2,1]**2 + matrix[2,2]**2))
        cp = np.cos(p)
        if not np.isclose(cp, 0.0):
            o = np.arctan2(-matrix[2,1], matrix[2,2])
            k = np.arctan2(-matrix[1,0], matrix[0,0])
        else:
            o = 0.0
            k = np.arctan2(matrix[0,1], matrix[0,2])
            if not np.isclose(p, np.pi/2):
                k *= -1
        return (
            np.rad2deg(o),
            np.rad2deg(p),
            np.rad2deg(k)
        )

    @staticmethod
    def _rot_wgs84_metric(o, p, k, meridian_convergence):
        r_pose = GeoFrame._opk_to_matrix(o, p, k)
        r_meridian_convergence = GeoFrame._opk_to_matrix(0, 0, np.deg2rad(meridian_convergence))
        r_mc_T = r_meridian_convergence.T
        r = r_mc_T @ r_pose
        return GeoFrame._matrix_to_opk(r.T)

    @staticmethod
    def apply_matrix4(matrix4x4, points):
        points = np.atleast_2d(points)
        if points.shape[1] != 3:
            raise ValueError("Input points must be of shape (3,) or (N, 3)")
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])
        transformed = (matrix4x4 @ points_homogeneous.T).T
        w = transformed[:, 3:4]
        nonzero_w = np.where(w == 0, 1, w)
        transformed_xyz = transformed[:, :3] / nonzero_w
        return transformed_xyz[0] if transformed_xyz.shape[0] == 1 else transformed_xyz
    

import numpy as np

def ypr2mat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Reproduce Metashape's ypr2mat behavior.
    Args:
        yaw (float): Rotation about Y-axis (degrees).
        pitch (float): Rotation about X-axis (degrees).
        roll (float): Rotation about Z-axis (degrees).
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    # Convert degrees to radians
    yaw = -np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # Rotation matrices for each axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    R_y = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combine rotations: Yaw → Pitch → Roll
    R = R_z @ R_x @ R_y 
    return R

def transform_to_camera_crs(x: np.ndarray, y: np.ndarray, z: np.ndarray, camera_meta: dict) -> tuple:
    """
    Transforms points from local CRS to camera CRS.
    
    Args:
        x, y, z: 1D arrays (NumPy arrays, or Pandas Series) of 
                 the x, y, z coordinates of points in local CRS.
        camera_pose: Dictionary containing the camera orientation:
                     {'yaw': yaw, 'pitch': pitch, 'roll': roll}
                     
    Returns:
        A Tuple of three 1D arrays (x', y', z') representing points in camera CRS.
    """

    # translation  
    x = np.array(x - camera_meta['x'])
    y = np.array(y - camera_meta['y'])
    z = np.array(z - camera_meta['z'])

    # Rotation 
    R = ypr2mat(camera_meta['yaw'], camera_meta['pitch'], camera_meta['roll'])  # Camera to local
    R_bore = ypr2mat(0, -90, 0)  # Boresight to camera
    R = R @ R_bore.T @ np.diag(np.array([1, -1, -1]))
     
    # Stack points for vectorized transformation
    points = np.stack([x, y, z], axis=-1)  # shape (N, 3)
    transformed_points = (R.T @ points.T).T  # shape (N, 3)
    x_cam, y_cam, z_cam = transformed_points.T
    
    return x_cam, y_cam, z_cam

def spherical_projection(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: int, h: int) -> tuple:
    """
    Project 3D LiDAR points to 2D spherical coordinates.

    Args:
        x (np.ndarray): X coordinates of the points.
        y (np.ndarray): Y coordinates of the points.
        z (np.ndarray): Z coordinates of the points.
        w (int): Width of the output image.
        h (int): Height of the output image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            u (np.ndarray): U coordinates in the image.
            v (np.ndarray): V coordinates in the image.
    """
    # Map azimuth and elevation to image coordinates
    f = w / (2 * np.pi) 
    u = 0.5 * w + f * np.arctan2(x, z)  
    v = 0.5 * h + f * np.arctan2(y, np.sqrt(x**2 + z**2)) 

    return u, v

def spherical_unprojection(u: np.ndarray, v: np.ndarray, r: np.ndarray, w: int, h: int) -> tuple:
    """
    Reverse spherical projection: map 2D pixel coordinates (u, v) and depth r
    back to 3D camera CRS coordinates (x, y, z).

    Args:
        u (np.ndarray): U coordinates in the image.
        v (np.ndarray): V coordinates in the image.
        r (np.ndarray): Depth distances of the points.
        w (int): Width of the image.
        h (int): Height of the image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            x (np.ndarray): X coordinates in camera CRS.
            y (np.ndarray): Y coordinates in camera CRS.
            z (np.ndarray): Z coordinates in camera CRS.
    """
    f = w / (2 * np.pi)
    # Inverse mapping from pixel to spherical angles
    azimuth = (u - 0.5 * w) / f  # theta
    elevation = (v - 0.5 * h) / f  # phi

    # Spherical to Cartesian (camera CRS)
    x = r * np.sin(azimuth) * np.cos(elevation)
    y = r * np.sin(elevation)
    z = r * np.cos(azimuth) * np.cos(elevation)
    return x, y, z

def transform_to_local_crs(x_cam: np.ndarray, y_cam: np.ndarray, z_cam: np.ndarray, camera_meta: dict) -> tuple:
    """
    Transforms points from camera CRS back to local CRS.

    Args:
        x_cam, y_cam, z_cam: 1D arrays of coordinates in camera CRS.
        camera_ori: Dictionary containing the camera orientation:
                    {'yaw': yaw, 'pitch': pitch, 'roll': roll}

    Returns:
        Tuple of three 1D arrays (x, y, z) in local CRS.
    """
    # Rotation 
    R = ypr2mat(camera_meta['yaw'], camera_meta['pitch'], camera_meta['roll'])  # Camera to local
    R_bore = ypr2mat(0, -90, 0)  # Boresight to camera
    R = R @ R_bore.T @ np.diag(np.array([1, -1, -1]))

    # Stack points for vectorized transformation
    points = np.stack([x_cam, y_cam, z_cam], axis=-1)  # shape (N, 3)
    transformed_points = (R @ points.T).T  # shape (N, 3)
    x, y, z = transformed_points.T

    # translation  
    x += camera_meta['x']
    y += camera_meta['y']
    z += camera_meta['z']
    
    return x, y, z

def euler_zyx_to_matrix(rz, ry, rx):
    """Build rotation matrix from intrinsic ZYX (yaw-pitch-roll) angles."""
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1]
    ])

    Ry = np.array([
        [cy, 0, sy],
        [ 0, 1,  0],
        [-sy, 0, cy]
    ])

    Rx = np.array([
        [1,  0,   0],
        [0, cx, -sx],
        [0, sx,  cx]
    ])

    return Rz @ Ry @ Rx

def rotation_matrix_to_opk(R):
    """
    Extract omega (X), phi (Y), kappa (Z) from rotation matrix.
    Assumes world to camera rotation (ENU to right-up-backward).
    """
    
    if abs(R[2, 0]) < 1.0:
        phi = np.arcsin(-R[2, 0])
        omega = np.arctan2(R[2,1] / np.cos(phi), R[2,2] / np.cos(phi))
        kappa = np.arctan2(R[1,0] / np.cos(phi), R[0,0] / np.cos(phi))
    else:
        # Gimbal lock
        phi = np.pi / 2 if R[2,0] <= -1.0 else -np.pi / 2
        omega = 0
        kappa = np.arctan2(-R[0,1], R[1,1])
    
    return omega, phi, kappa

def rxryrz_to_opk(rx, ry, rz):
    R = euler_zyx_to_matrix(rz, ry, rx)  # Note: RzRyRx intrinsic order
    return rotation_matrix_to_opk(R)



