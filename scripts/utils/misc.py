import sys
import json
import pygeohash as pgh
import networkx as nx

from loguru import logger
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import explain_validity, make_valid


def assign_groups(row, group_index):
    """Assign a group number to GT and detection of a geodataframe

    Args:
        row (row): geodataframe row

    Returns:
        row (row): row with a new 'group_id' column
    """

    try:
        row['group_id'] = group_index[row['geohash_left']]
    except: 
        row['group_id'] = None
    
    return row
    


def format_logger(logger):

    logger.remove()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO", filter=lambda record: record["level"].no < 25)
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - <green>{level}</green> - {message}",
            level="SUCCESS", filter=lambda record: record["level"].no < 30)
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - <yellow>{level}</yellow> - {message}",
            level="WARNING", filter=lambda record: record["level"].no < 40)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <red>{level}</red> - <level>{message}</level>",
            level="ERROR")
    
    return logger


def find_category(df):

    if 'category' in df.columns:
        df.rename(columns={'category': 'CATEGORY'}, inplace = True)
    elif 'CATEGORY' not in df.columns:
        logger.critical('The GT labels have no category. Please produce a CATEGORY column when preparing the data.')
        sys.exit(1)

    if 'supercategory' in df.columns:
        df.rename(columns={'supercategory': 'SUPERCATEGORY'}, inplace = True)
    elif 'SUPERCATEGORY' not in df.columns:
        logger.critical('The GT labels have no supercategory. Please produce a SUPERCATEGORY column when preparing the data.')
        sys.exit(1)
    
    return df


def geohash(row):
    """Geohash encoding (https://en.wikipedia.org/wiki/Geohash) of a location (point).
    If geometry type is a point then (x, y) coordinates of the point are considered. 
    If geometry type is a polygon then (x, y) coordinates of the polygon centroid are considered. 
    Other geometries are not handled at the moment    

    Args:
        row: geodaframe row

    Raises:
        Error: geometry error

    Returns:
        out (str): geohash code for a given geometry
    """
    
    if row.geometry.geom_type == 'Point':
        out = pgh.encode(latitude=row.geometry.y, longitude=row.geometry.x, precision=16)
    elif row.geometry.geom_type == 'Polygon':
        out = pgh.encode(latitude=row.geometry.centroid.y, longitude=row.geometry.centroid.x, precision=16)
    else:
        logger.error(f"{row.geometry.geom_type} type is not handled (only Point or Polygon geometry type)")
        sys.exit()

    return out


def get_number_of_classes(coco_files_dict):
    """Read the number of classes from the tileset COCO file.

    Args:
        coco_files_dict (dict): COCO file of the tileset

    Returns:
        num_classes (int): number of classes in the dataset
    """

    file_content = open(next(iter(coco_files_dict.values())))
    coco_json = json.load(file_content)
    num_classes = len(coco_json["categories"])
    file_content.close()
    if num_classes == 0:
        logger.critical('No defined class in the 1st COCO file.')
        sys.exit(0)

    logger.info(f"Working with {num_classes} class{'es' if num_classes > 1 else ''}.")

    return num_classes


def make_groups(gdf):
    """Identify groups based on pairing nodes with NetworkX. The Graph is a collection of nodes.
    Nodes are hashable objects (geohash (str)).

    Returns:
        groups (list): list of connected geohash groups
    """

    g = nx.Graph()
    for row in gdf[gdf.geohash_left.notnull()].itertuples():
        g.add_edge(row.geohash_left, row.geohash_right)

    groups = list(nx.connected_components(g))

    return groups


def segmentation_to_polygon(segm):
    # transform segmentation coordinates to a polygon or a multipolygon
    if len(segm)==1:
        if len(segm[0])<5:
            return None
        x = segm[0][0::2]
        y = segm[0][1::2]
        poly = Polygon(zip(x, y))
    else:
        parts = []
        for coord_list in segm:
            if len(coord_list)<5:
                    continue
            x = coord_list[0::2]
            y = coord_list[1::2]
            parts.append(Polygon(zip(x, y)))
        if len(parts)==0:
            return None
        poly = MultiPolygon(parts) if len(parts)>1 else parts[0]

    if not poly.is_valid and 'Self-intersection' in explain_validity(poly):
        valid_poly = make_valid(poly)
        if isinstance(valid_poly, GeometryCollection):
            tmp_list = []
            for multi_geom in valid_poly.geoms:
                if isinstance(multi_geom, MultiPolygon):
                    tmp_list.extend([geom for geom in multi_geom.geoms])
                else:
                    tmp_list.extend([multi_geom])
            poly = MultiPolygon([geom for geom in tmp_list if isinstance(geom, Polygon)])
            if poly.is_empty:
                poly = MultiPolygon([geom for geom in valid_poly.geoms if isinstance(geom, Polygon)])
        else:
            poly = valid_poly

    if not poly.is_valid:
        logger.warning(f"Polygon is not valid: {poly}")

    if poly.area == 0:
        logger.warning(f"Polygon area is 0: {poly}")

    return poly