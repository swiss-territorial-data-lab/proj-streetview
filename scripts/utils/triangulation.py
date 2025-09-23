"""
Triangulation utilities for projecting 2D detections to 3D and aggregating them
across multiple frames. Rays are built from per-frame detections, pairwise
closest-approach intersections are computed, and a peeling strategy clusters
these intersections into stable 3D candidates.

Public entry points:
- cylin_pano_proj_ray / cubemap_pano_proj_ray: build rays from detections.
- triangulation_peeling: main routine for clustering intersections into candidates.
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon
from typing import List, Dict
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from .projection import *
from collections import defaultdict
from sklearn.neighbors import KDTree
from skimage.measure import EllipseModel


class Ray():
    """A ray in 3D associated with a frame and an optional detection.

    Attributes:
        origin (np.ndarray): Camera center in world coordinates (3,).
        direction (np.ndarray): Unit direction vector from origin (3,).
        frame_id (int): Frame index used by the triangulation loop.
        candidate_id (Optional[int]): Linked candidate id if assigned.
        mask_area (Optional[float]): Pixel area for optional size-consistency checks.
    """
    def __init__(self, origin: np.ndarray, direction: np.ndarray, frame_id: int, candidate_id=None, mask_area=None):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)
        self.frame_id = frame_id
        self.candidate_id = candidate_id  # which candidate this ray is currently associated with
        self.mask_area = mask_area


# Intersection class definition
class Intersection:
    """Metadata for the closest-approach between two rays.

    Args:
        point (np.ndarray): Midpoint between the two rays at closest approach (3,).
        ray_pair (Tuple[int, int]): Indices of the two rays in the global ray list.
        dist (float): Separation between the two rays at closest approach (m).
        length (float): Max forward distance from ray origins to closest point (m).
    """
    def __init__(self, point, ray_pair, dist, length):
        self.point = point          # np.array([x, y, z])
        self.ray_pair = ray_pair    # tuple of (prev_idx, new_idx)
        self.dist = dist            # float, distance between rays at intersection
        self.length = length        # float, sum of distances from ray origins to intersection


# Candidate class for storing candidate attributes
class Candidate:
    """Evolving 3D candidate aggregated from multiple intersections.

    Attributes:
        center (np.ndarray): Robust center estimate (median) in world coordinates.
        intersections (Set[int]): Contributing intersection indices.
        mean_intersection_length (float): Mean forward distance of intersections.
        mean_intersection_dist (float): Mean closest-approach distance (quality metric).
        last_seen (int): Frame index when last updated.
        missing (int): Consecutive frames without an update.
    """
    def __init__(self, center, intersections, mean_intersection_length, mean_intersection_dist, last_seen, missing):
        self.center = center
        self.intersections = set(intersections)
        self.mean_intersection_length = mean_intersection_length
        self.mean_intersection_dist = mean_intersection_dist
        self.last_seen = last_seen
        self.missing = missing


    def update(self, center, intersections, mean_intersection_length, mean_intersection_dist, last_seen):
        """Merge in intersections and refresh center; reset missing counter."""
        self.center = center
        self.intersections = set(intersections)
        self.mean_intersection_length = mean_intersection_length
        self.mean_intersection_dist = mean_intersection_dist
        self.last_seen = last_seen
        self.missing = 0


    def increment_missing(self):
        """Mark that this candidate was not updated in the current frame."""
        self.missing += 1


    def to_row(self):
        """Serialize to a GeoDataFrame row dictionary."""
        return {
            'elevation': self.center[2],
            'intersections': len(self.intersections),
            'mean_intersection_length': self.mean_intersection_length,
            'mean_intersection_dist': self.mean_intersection_dist,
            'geometry': Point(self.center[0], self.center[1])
        }


def manhole_radius_from_pixel_area(mask_area, ray_dir, Z, W, H, manhole_normal=[0,1,0]):
    """
    Invert the ellipse projection: given pixel-area bounds in panorama,
    compute corresponding physical manhole radius bounds.

    Parameters
    ----------
    A_px_min, A_px_max : float
        Pixel-area bounds in the panorama.
    ray_dir : array_like, shape (3,)
        Direction vector from camera to manhole center.
    Z : float
        Distance from camera to manhole center [m].
    W, H : int
        Panorama width and height in pixels.
    manhole_normal : array_like, shape (3,)
        Unit normal vector of manhole disk (default up).

    Returns
    -------
    R: float
        Physical manhole radius bounds [m].
    """
    ray_dir = np.asarray(ray_dir, dtype=float)
    if np.linalg.norm(ray_dir) == 0:
        raise ValueError("ray_dir must be non-zero")
    vhat = ray_dir / np.linalg.norm(ray_dir)

    n_hat = np.asarray(manhole_normal, dtype=float)
    n_hat /= np.linalg.norm(n_hat)

    cos_gamma = abs(np.dot(n_hat, vhat))
    vy = vhat[1]
    cos_phi = np.sqrt(max(1.0 - vy**2, 1e-12))  # avoid division by zero

    scale = (W * H) / (2 * np.pi**2) * (cos_gamma / cos_phi)

    R = Z * np.sqrt(mask_area / scale)

    return R


def project_ellipse_center_to_world(camera_meta: np.ndarray, ellipse_center_px: np.ndarray) -> Dict:
    """
    Project ellipse center in image to a ray in world coordinates.
    Returns {'origin': np.ndarray, 'direction': np.ndarray}
    """
    
    film_coords = spherical_unprojection(ellipse_center_px[0], ellipse_center_px[1], 1, camera_meta['width'], camera_meta['height'])
    x_local, y_local, z_local = transform_to_local_crs(film_coords[0], film_coords[1], film_coords[2], camera_meta)
    ray_vertex = np.array([x_local, y_local, z_local])

    cam_center = np.array([camera_meta['x'], camera_meta['y'], camera_meta['z']])
    ray_ori = ray_vertex - cam_center
    ray_data = {
        'origin': cam_center,
        'direction': ray_ori
    }
    return ray_data


def cubemap_pixel_to_world(cubemap: GeoFrame, pixel_coords: np.ndarray, face_idx: int) -> Dict:
    """
    Project ellipse center in image to a ray in world coordinates.
    Returns {'origin': np.ndarray, 'direction': np.ndarray}
    """

    if len(pixel_coords.shape) == 1:
        pixel_coords = pixel_coords[None,:]

    if pixel_coords.shape[1] != 2:
        print(f"ValueError: pixel_coords must have shape (?, 2), but got {pixel_coords.shape}")

    image_coords = cubemap.sensor_to_image(pixel_coords)
    model_coords = cubemap.image_to_model(image_coords)
    frame_coords, sid = cubemap.model_to_frame(model_coords, face_idx)
    world_coords = cubemap.frame_to_world(frame_coords)
    ray_vertex = world_coords.flatten()
    cam_center = cubemap.get_world_ori_for_sensor_id(*sid)[:3]
    ray_ori = ray_vertex - cam_center
    ray_data = {
        'origin': np.array(cam_center),
        'direction': np.array(ray_ori)
    }
    return ray_data


def spatial_temporal_group_sort(
    gdf,
    groupby_col='image_id',
    time_column="gps_sec_s_",
    x_col="x_m_",
    y_col="y_m_",
    z_col="z_m_",
    radius=10.0,
    output_column="sort_index"
):
    """
    Two-stage spatial-temporal sort on groupby object.
    
    Parameters:
        gdf: GroupBy GeoDataFrame.
        groupby_col: GroupBy column.
        time_column (str): Column for GPS time (shared within each group).
        x_col, y_col, z_col (str): Position columns (shared within each group).
        radius (float): Search radius in meters.
        output_column (str): Name of output sort index column.
    
    Returns:
        GeoDataFrame: With new sort_index column.
        GroupBy: Sorted groupby object (ordered by sort_index).
    """

    # Step 1: Extract one row per group to represent its position and time
    gdf_grouped = gdf.groupby(groupby_col)
    group_meta = (
        gdf_grouped[[time_column, x_col, y_col, z_col]]
        .first()
        .reset_index()
    )

    # Sort groups by time (Stage 1)
    group_meta = group_meta.sort_values(by=time_column).reset_index(drop=True)
    positions = group_meta[[x_col, y_col, z_col]].values
    image_ids = group_meta[groupby_col].values

    tree = KDTree(positions)
    used = np.zeros(len(group_meta), dtype=bool)
    final_order_indices = []

    prev_anchor_idx = None

    # sort groups by position (Stage 2)
    # Use temporal sorted camera as anchor 
    # extract all camera near the anchor in a given radius and sort spatially 
    # according to temporal trajectory (from previous anchor to current one)
    for current_anchor_idx in range(len(group_meta)):
        if used[current_anchor_idx]:
            continue

        current_pos = positions[current_anchor_idx]

        if prev_anchor_idx is None:
            group_indices = [current_anchor_idx]
        else:
            prev_pos = positions[prev_anchor_idx]
            direction = current_pos - prev_pos
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-6:
                direction = np.array([1, 0, 0])
            else:
                direction = direction / direction_norm

            neighbor_indices = tree.query_radius([current_pos], r=radius)[0]
            neighbor_indices = [i for i in neighbor_indices if not used[i]]

            if not neighbor_indices:
                continue

            neighbor_positions = positions[neighbor_indices]
            projections = neighbor_positions @ direction
            sorted_local = np.array(neighbor_indices)[np.argsort(projections)]
            group_indices = sorted_local.tolist()

        used[group_indices] = True
        final_order_indices.extend(group_indices)
        prev_anchor_idx = current_anchor_idx

    # Create mapping from image_id to sort index
    sort_index_map = {image_ids[idx]: i for i, idx in enumerate(final_order_indices)}

    # Apply mapping to original GeoDataFrame
    gdf_with_sort = gdf_grouped.obj.copy()
    gdf_with_sort[output_column] = gdf_with_sort[groupby_col].map(sort_index_map)

    # Return new dataframe and sorted groupby object
    gdf_sorted = gdf_with_sort.sort_values(by=output_column)
    return gdf_sorted.groupby(groupby_col, sort=False)


def compute_ray_intersection(ray1: Ray, ray2: Ray, radius: float = 10.0, distance_threshold: float = 0.5, ac: bool=False, hc: bool=True) -> np.ndarray:
    """
    Return intersection point if rays intersect in their forward direction and are close enough.
    Otherwise, return None.
    """
    p1, d1 = ray1.origin, ray1.direction
    p2, d2 = ray2.origin, ray2.direction

    v12 = p2 - p1
    d1_dot_d2 = np.dot(d1, d2)
    denom = 1 - d1_dot_d2 ** 2
    if abs(denom) < 1e-6:
        return None, None, None  # Nearly parallel, no intersection in forward direction
    t1 = (np.dot(v12, d1) - np.dot(v12, d2) * d1_dot_d2) / denom
    t2 = - (np.dot(v12, d2) - np.dot(v12, d1) * d1_dot_d2) / denom
    if t1 < 0 or t2 < 0 or t1 > radius or t2 > radius:
        return None, None, None # Closest approach is behind at least one ray's origin

    # whether deploy mask area control
    if ac:
        r1 = manhole_radius_from_pixel_area(ray1.mask_area, ray1.direction, t1, W=8000, H=4000)
        r2 = manhole_radius_from_pixel_area(ray2.mask_area, ray2.direction, t2, W=8000, H=4000)
        
        if np.abs(r2-r1) > 0.2:
            return None, None, None

    point1 = p1 + t1 * d1
    point2 = p2 + t2 * d2
    dist = np.linalg.norm(point1 - point2)

    if dist > distance_threshold:
        return None, None, None  # Closest points are too far apart to be considered an intersection
    
    intersect = (point1 + point2) / 2
    if hc:
        if intersect[2] >= min(p1[2], p2[2]) - 2.0 or intersect[2] <= min(p1[2], p2[2]) - 3.8:
            return None, None, None
    return intersect, dist, max(t1, t2)


def cluster_intersections(intersections: List[np.ndarray], distance_threshold: float = 0.5) -> List[List[int]]:
    """Cluster intersection points by spatial proximity using DBSCAN."""
    if len(intersections) == 0:
        return []
    X = np.stack(intersections)
    db = DBSCAN(eps=distance_threshold, min_samples=1).fit(X)
    labels = db.labels_
    clusters = []
    for label in set(labels):
        cluster = np.where(labels == label)[0].tolist()
        clusters.append(cluster)
    return clusters


def load_coco_inferences(coco_json_path: str, t_score: float=0.5) -> gpd.GeoDataFrame:
    """Load COCO-style detections and return image and annotation tables.

    Args:
        coco_json_path (str): Path to a COCO JSON file containing images and annotations.
        t_score (float): Keep annotations with score >= t_score.

    Returns:
        Tuple[pd.DataFrame, gpd.GeoDataFrame]: images dataframe and annotations GeoDataFrame.
    """
    # Load COCO data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Build a DataFrame from coco_data['images']
    images_df = pd.DataFrame(coco_data['images'])
    images_df.id = images_df.id.astype(int)

    # Prepare a list of annotation records
    records = []
    for ann in coco_data['annotations']:
        if float(ann['score']) < t_score:
            continue
        image_id = ann['image_id']
        segmentation = ann['segmentation'][0]

        coords = [(segmentation[i], segmentation[i+1]) for i in range(0, len(segmentation), 2)]
        geometry = Polygon(coords)
        records.append({
            'image_id': image_id,
            'category_id': ann.get('category_id', None),
            'geometry': geometry,
            'annotation_id': ann.get('id', None),
            'score': ann.get('score', None),
            'area': ann.get('area', None)
        })

    ann_gdf = gpd.GeoDataFrame(records, geometry='geometry', crs="epsg:2056")

    return images_df, ann_gdf


# Extract '102-35582' as frame_id, 'lb4' as camera_model, and '0' as cube_idx from '102-lb4-0-35582.jpg'
def parse_file_name(file_name):
    """Parse file names like '102-lb4-0-35582.jpg' to components.

    Returns a pandas Series with 'frame_id', 'camera_model', and 'cube_idx'.
    """
    # Remove extension
    base = file_name.replace('.jpg', '')
    # Split by '-'
    parts = base.split('-')
    if len(parts) < 4:
        return pd.Series({'frame_id': None, 'camera_model': None, 'cube_idx': None})
    camera_model = parts[1]
    cube_idx = parts[2]
    frame_id = f"{parts[0]}-{parts[3]}"
    return pd.Series({'frame_id': frame_id, 'camera_model': camera_model, 'cube_idx': cube_idx})


def cylin_pano_proj_ray(
    frame_id, 
    frame, 
    meta_cols=['x_m_', 'y_m_', 'z_m_','gpsimgdirection', 'gpspitch', 'gpsroll'], 
    offset=[0, 0, 0, 0, 0, 0]
    ): 
    """Construct rays for cylindrical panoramas using ellipse-fitted detection centers."""
    _, df_frame = frame
    camera_meta = {
        'x':        df_frame.iloc[0][meta_cols[0]] + offset[0],
        'y':        df_frame.iloc[0][meta_cols[1]] + offset[1],
        'z':        df_frame.iloc[0][meta_cols[2]] + offset[2],
        'yaw':      df_frame.iloc[0][meta_cols[3]] + offset[3],       
        'pitch':    df_frame.iloc[0][meta_cols[4]] + offset[4],
        'roll':     df_frame.iloc[0][meta_cols[5]] + offset[5],             
        'width':    df_frame.iloc[0]['width'],
        'height':   df_frame.iloc[0]['height']
    }

    # Get predicted masks for this image 
    new_rays = []
    for _, det in df_frame.iterrows():
        # Fit a minimum area ellipse to the contour of det.geometry and use its center as center_px
        coords = np.array(det.geometry.exterior.coords)
        ellipse = EllipseModel()
        success = ellipse.estimate(coords)
        if success:
            xc, yc, a, b, theta = ellipse.params
            center_px = (xc, yc)
        else:
            # Fallback to centroid if ellipse fit fails
            center_px = np.array(det.geometry_y.centroid.coords)[0]
        ray_data = project_ellipse_center_to_world(camera_meta, center_px)
        ray = Ray(ray_data['origin'], ray_data['direction'], frame_id, mask_area=det.area)
        new_rays.append(ray)

    return new_rays


def cubemap_pano_proj_ray(
    frame_id, 
    frame, 
    meta_cols=['x', 'y', 'z','rx', 'ry', 'rz', 'camera_model', 'size'], 
    offset=[0, 0, 0, 0, 0, 0]
    ):
    """Construct rays for cube-map panoramas using ellipse-fitted detection centers.

    Filters poor ellipse fits by IoU with the original polygon before projection.
    """
    _, df_frame = frame
    cam_params = df_frame.iloc[0][meta_cols]
    # Create a GeoFrame instance with the provided metadata
    imagemeta = ImageMeta(
        width=cam_params[meta_cols[7]],
        height=cam_params[meta_cols[7]],
        pixsize=0.001,
        focal_length=float(cam_params[meta_cols[7]] * 0.001 / 2)
    )

    omega, phi, kappa = rxryrz_to_opk(
        cam_params[meta_cols[3]] + offset[3], 
        cam_params[meta_cols[4]] + offset[4], 
        cam_params[meta_cols[5]] + offset[5]
        )

    geoframe = GeoFrame(
        easting=cam_params[meta_cols[0]] + offset[0],
        northing=cam_params[meta_cols[1]] + offset[1],
        height=cam_params[meta_cols[2]] + offset[2],
        omega=omega,
        phi=phi,
        kappa=kappa,
        imagemeta=imagemeta,
        camera_model=cam_params[meta_cols[6]]
    )

    # Get predicted masks for this image 
    new_rays = []
    for _, det in df_frame.iterrows():
        # Fit a minimum area ellipse to the contour of det.geometry and use its center as center_px
        coords = np.array(det.geometry.exterior.coords)
        ellipse = EllipseModel()
        success = ellipse.estimate(coords)
        if success:
            xc, yc, a, b, theta = ellipse.params
            center_px = np.array([xc, yc])
            # Calculate IoU between fitted ellipse and det.geometry
            # Create a polygon approximation of the fitted ellipse
            t = np.linspace(0, 2 * np.pi, 100)
            ellipse_x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
            ellipse_y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
            ellipse_poly = Polygon(np.column_stack([ellipse_x, ellipse_y]))
            try:
                # det.geometry is assumed to be a Polygon
                intersection = ellipse_poly.intersection(det.geometry)
                union = ellipse_poly.union(det.geometry)
            except:
                continue
            iou = 0.0
            if not intersection.is_empty and not union.is_empty:
                iou = intersection.area / union.area

            if iou < 0.85:
                continue
        else:
            continue
        ray_data = cubemap_pixel_to_world(geoframe, center_px, int(det.cube_idx))
        ray = Ray(ray_data['origin'], ray_data['direction'], frame_id, mask_area=det.area)
        new_rays.append(ray)

    return new_rays


def triangulation_peeling(
    grouped,
    pano_proj_ray,                       
    intersection_threshold=0.2,  
    clustering_threshold=0.5,             # meters, for intersection and clustering
    candidate_update_threshold=1.0,       # meters, for candidate consistency
    candidate_missing_limit=5,            # number of consecutive frames without new rays before removal
    radius=20.0,
    mask_area_control=False,
    height_control=True,
    tqdm_desc="Processing frames"
):
    """
    Perform triangulation and candidate peeling clustering on grouped frames.

    Args:
        grouped: iterable of (image_id, df_frame) pairs, grouped and sorted.
        pano_proj_ray: function receive (frame_id, frame, meta_cols, offset), create 3d ray using defined projection
        intersection_threshold: float, distance threshold for valid ray intersection.
        clustering_threshold: float, DBSCAN eps for intersection/candidate clustering.
        candidate_update_threshold: float, max distance to update candidate.
        candidate_missing_limit: int, max missing frames before candidate removal.
        radius: float, max distance for ray intersection.
        mask_area_control: bool, use mask area in intersection.
        height_control: bool, use height in intersection.
        tqdm_desc: str, description for tqdm progress bar.

    Returns:
        out_gdf: GeoDataFrame of final candidates.
        candidate_list: list of Candidate objects.
        ray_list: list of Ray objects.
        intersection_objs: list of Intersection objects.
    """
    # State
    ray_list = []  # List of Ray objects (never removed, indices are constant)
    candidate_list = []  # List of Candidate objects
    intersection_objs = []  # List of Intersection objects

    iterator = tqdm(grouped, total=len(grouped), desc=tqdm_desc)

    # Maintain sets of active indices for rays, candidates, and intersections
    active_ray_indices = set()
    active_candidate_indices = set()
    active_intersection_indices = set()

    false_pos_indices = set()

    for frame_id, frame in enumerate(iterator):

        new_rays = pano_proj_ray(frame_id, frame)

        # Add new rays to the ray list
        ray_start_idx = len(ray_list)
        ray_list.extend(new_rays)
        new_ray_indices = np.arange(ray_start_idx, len(ray_list))

        n_prev_rays = len(active_ray_indices)
        # When new rays are added, mark their indices as active
        active_ray_indices.update(new_ray_indices)
        # Compute intersections only between new rays and previous active rays, but store all intersections globally
        n_new_rays = len(new_rays)
        new_intersection_indices = []
        if n_prev_rays > 0 and n_new_rays > 0:
            for new_idx, ray_new in zip(new_ray_indices, new_rays):
                best_intersections = dict()
                for prev_idx in active_ray_indices:
                    ray_prev = ray_list[prev_idx]
                    # Only consider pairs from different frames
                    if ray_prev.frame_id != ray_new.frame_id:
                        pt, dist, length = compute_ray_intersection(
                            ray_prev, 
                            ray_new, 
                            radius=radius, 
                            distance_threshold=intersection_threshold, 
                            ac=mask_area_control,
                            hc=height_control)
                        if pt is not None:
                            frame_pair = tuple(sorted([ray_prev.frame_id, ray_new.frame_id]))
                            if frame_pair not in best_intersections or dist < best_intersections[frame_pair].dist:
                                best_intersections[frame_pair] = Intersection(
                                    point=pt,
                                    ray_pair=(prev_idx, new_idx),
                                    dist=dist,
                                    length=length
                                )
                # After collecting, add only the best intersections for each frame pair
                for intersection in best_intersections.values():
                    intersection_objs.append(intersection)
                    new_intersection_indices.append(len(intersection_objs) - 1)
        # Add new intersections to active set
        active_intersection_indices.update(new_intersection_indices)

        # Peeling clustering: iteratively select the largest spatial cluster, then
        # remove any intersection that shares a ray with that cluster. This helps
        # suppress ambiguous hypotheses and keeps one coherent set per iteration.
        clusters_global = []
        if len(active_intersection_indices) > 0:
            remaining_indices = set(active_intersection_indices)
            while len(remaining_indices) > 0:
                intersection_points = [intersection_objs[i].point for i in remaining_indices]
                X = np.stack(intersection_points)
                db = DBSCAN(eps=clustering_threshold, min_samples=1).fit(X)
                labels = db.labels_
                unique_labels, counts = np.unique(labels, return_counts=True)
                largest_label = unique_labels[np.argmax(counts)]
                largest_cluster_local_indices = np.where(labels == largest_label)[0]
                remaining_indices_list = list(remaining_indices)
                largest_cluster_global_indices = [remaining_indices_list[i] for i in largest_cluster_local_indices]
                clusters_global.append(largest_cluster_global_indices)
                rays_in_largest_cluster = set()
                for idx in largest_cluster_global_indices:
                    rays_in_largest_cluster.update(intersection_objs[idx].ray_pair)
                to_remove = set()
                for idx in remaining_indices:
                    if len(set(intersection_objs[idx].ray_pair) & rays_in_largest_cluster) > 0:
                        to_remove.add(idx)
                # record only the OTHER intersections (not the current largest cluster itself)
                fp_idx = to_remove.difference(set(largest_cluster_global_indices))
                false_pos_indices.update(fp_idx)
                # remove all marked intersections from active/remaining (both cluster and others)
                remaining_indices.difference_update(to_remove)
                active_intersection_indices.difference_update(fp_idx)

        # For each cluster, use the median of intersections as a robust candidate center
        new_candidates = []
        if len(clusters_global) > 0:
            for cluster in clusters_global:
                cand_inters = [intersection_objs[i] for i in cluster]
                all_points = np.array([inter.point for inter in cand_inters])
                all_indices = set(cluster)
                all_lengths = [inter.length for inter in cand_inters]
                all_dists = [inter.dist for inter in cand_inters]
                mean_intersection_length = float(np.mean(all_lengths)) if all_lengths else float('nan')
                mean_intersection_dist = float(np.mean(all_dists)) if all_dists else float('nan')
                center = np.median(all_points, axis=0)
                new_candidates.append(
                    Candidate(
                        center=center,
                        intersections=all_indices,
                        mean_intersection_length=mean_intersection_length,
                        mean_intersection_dist=mean_intersection_dist,
                        last_seen=frame_id,
                        missing=0
                    )
                )

        # Update existing candidates or add new ones (dynamic pool)
        updated_candidate_ids = set()
        for nc in new_candidates:
            nc_center = nc.center
            nc_inters = nc.intersections
            nc_mean_length = nc.mean_intersection_length
            nc_mean_dist = nc.mean_intersection_dist
            matched = False
            active_candidate_indices_list = list(active_candidate_indices)

            if len(active_candidate_indices) > 0:
                candidate_centers = np.array([candidate_list[cid].center for cid in active_candidate_indices_list])
                # candidate_centers = np.array([cand.center for cand in candidate_list[active_candidate_indices]])
                dists = np.linalg.norm(candidate_centers - nc_center, axis=1)
                within_radius_indices = np.where(dists < radius)[0]
                if len(within_radius_indices) > 0:
                    for cid, cand in [(int(cid), candidate_list[int(cid)]) for cid in within_radius_indices.tolist()]:
                        dist = np.linalg.norm(nc_center - cand.center)
                        if dist < candidate_update_threshold and cand.intersections != nc_inters:
                            combined_inters = cand.intersections.union(nc_inters)
                            
                            active_intersection_indices.update(cand.intersections)
                            rays_to_activate = set()
                            for iid in combined_inters:
                                inter = intersection_objs[iid]
                                rays_to_activate.update(inter.ray_pair)
                            active_ray_indices.update(rays_to_activate)

                            pts = np.array([intersection_objs[i].point for i in combined_inters])
                            lengths_all = [intersection_objs[i].length for i in combined_inters]
                            dists_all = [intersection_objs[i].dist for i in combined_inters]
                            new_center = np.median(pts, axis=0)
                            mean_intersection_length = float(np.mean(lengths_all)) if lengths_all else float('nan')
                            mean_intersection_dist = float(np.mean(dists_all)) if dists_all else float('nan')
                            cand.update(
                                center=new_center,
                                intersections=set(combined_inters),
                                mean_intersection_length=mean_intersection_length,
                                mean_intersection_dist=mean_intersection_dist,
                                last_seen=frame_id
                            )
                            updated_candidate_ids.add(cid)
                            matched = True
                            break
                        elif cand.intersections == nc_inters:
                            matched = True
                            break

            if not matched:
                candidate_list.append(
                    Candidate(
                        center=nc_center,
                        intersections=set(nc_inters),
                        mean_intersection_length=nc_mean_length,
                        mean_intersection_dist=nc_mean_dist,
                        last_seen=frame_id,
                        missing=0
                    )
                )
                new_cid = len(candidate_list) - 1
                updated_candidate_ids.add(new_cid)
                active_candidate_indices.add(new_cid)

        # For candidates not updated, increment missing count
        for cid in list(active_candidate_indices):
            if cid not in updated_candidate_ids:
                cand = candidate_list[cid]
                cand.increment_missing()
                if cand.missing > candidate_missing_limit:
                    rays_to_remove = set()
                    for iid in cand.intersections:
                        inter = intersection_objs[iid]
                        rays_to_remove.update(inter.ray_pair)
                    active_ray_indices.difference_update(rays_to_remove)
                    
                    # Also deactivate intersections that only involve rays from this candidate
                    active_intersection_indices.difference_update(cand.intersections)
                    active_candidate_indices.discard(cid)

    # After temporal-spatial iteration with dynamic pool with active elements
    # Cluster all candidates inside candidate_update_threshold again
    # Duplicated detection results from overlap of trajectory and dynamic pool corner case will be merged

    # Prepare candidate centers for clustering
    candidate_centers = np.array([cand.center for cand in candidate_list])
    if len(candidate_centers) == 0:
        out_gdf = gpd.GeoDataFrame([], geometry='geometry', crs="epsg:2056")
    else:
        db = DBSCAN(eps=candidate_update_threshold, min_samples=1)
        # db = DBSCAN(eps=intersection_distance_threshold, min_samples=1)
        labels = db.fit_predict(candidate_centers)


        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        new_candidate_list = []
        old_to_new_cid = {}

        for label, indices in tqdm(clusters.items()):
            if len(indices) == 1:
                orig_cid = indices[0]
                new_cid = len(new_candidate_list)
                new_candidate_list.append(candidate_list[orig_cid])
                old_to_new_cid[orig_cid] = new_cid
            else:
                merged_intersections = set()
                merged_last_seen = -1
                merged_missing = float('inf')
                for cid in indices:
                    cand = candidate_list[cid]
                    merged_intersections.update(cand.intersections)
                    merged_last_seen = max(merged_last_seen, cand.last_seen)
                    merged_missing = min(merged_missing, cand.missing)
                intersection_points = []
                intersection_lengths = []
                intersection_dists = []
                for iid in merged_intersections:
                    intersection_points.append(intersection_objs[iid].point)
                    intersection_lengths.append(intersection_objs[iid].length)
                    intersection_dists.append(intersection_objs[iid].dist)
                if intersection_points:
                    new_center = np.median(np.array(intersection_points), axis=0)
                    mean_intersection_length = float(np.mean(intersection_lengths)) if intersection_lengths else float('nan')
                    mean_intersection_dist = float(np.mean(intersection_dists)) if intersection_dists else float('nan')
                    new_intersections = merged_intersections
                else:
                    centers = [candidate_list[k].center for k in indices]
                    new_center = np.median(np.array(centers), axis=0)
                    mean_intersection_length = float('nan')
                    mean_intersection_dist = float('nan')
                    new_intersections = set()
                new_cand = Candidate(
                    center=new_center,
                    intersections=new_intersections if new_intersections else merged_intersections,
                    mean_intersection_length=mean_intersection_length,
                    mean_intersection_dist=mean_intersection_dist,
                    last_seen=merged_last_seen,
                    missing=merged_missing
                )
                new_cid = len(new_candidate_list)
                for k in indices:
                    old_to_new_cid[k] = new_cid
                new_candidate_list.append(new_cand)

        candidate_list = new_candidate_list
        active_candidate_indices = set(old_to_new_cid[cid] for cid in active_candidate_indices if cid in old_to_new_cid)

        rows = []
        for cid, cand in enumerate(candidate_list):
            row = cand.to_row()
            rows.append(row)

        out_gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs="epsg:2056")

    
    # Filter out intersections flagged as false positives in the return value
    filtered_intersection_objs = [obj for i, obj in enumerate(intersection_objs) if i not in false_pos_indices]
    return out_gdf, candidate_list, ray_list, filtered_intersection_objs