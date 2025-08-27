import sys

from geopandas import GeoDataFrame
import networkx as nx
from shapely import Point, Polygon

from math import ceil

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
    """
    Configures the logger to format log messages with specific styles and colors based on their severity level.

    Args:
        logger (loguru.logger): The logger instance to be formatted.

    Returns:
        loguru.logger: The configured logger instance with custom formatting.
    """

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


def get_grid_size(tile_size, grid_width=256, grid_height=256, max_dx=0, max_dy=0):
    """Determine the number of grid cells based on the tile size, grid dimension and overlap between tiles.
    All values are in pixels.

    Args:
        tile_size (tuple): tile width and height
        grid_width (int, optional): width of a grid cell. Defaults to 256.
        grid_height (int, optional): height of a grid cell. Defaults to 256.
        max_dx (int, optional): overlap on the width. Defaults to 0.
        max_dy (int, optional): overlap on the height. Defaults to 0.

    Returns:
        number_cells_x: number of grid cells on the width
        number_cells_y: number of grid cells on the height
    """

    tile_width, tile_height = tile_size
    number_cells_x = ceil((tile_width - max_dx)/(grid_width - max_dx))
    number_cells_y = ceil((tile_height - max_dy)/(grid_height - max_dy))

    return number_cells_x, number_cells_y


def grid_over_tile(tile_size, tile_origin, pixel_size_x, pixel_size_y=None, max_dx=0, max_dy=0, grid_width=256, grid_height=256, crs='EPSG:2056', test_shape = None):
    """Create a grid over a tile and save it in a GeoDataFrame with each row representing a grid cell.

    Args:
        tile_size (tuple): tile width and height
        tile_origin (tuple): tile minimum coordinates
        pixel_size_x (float): size of the pixel in the x direction
        pixel_size_y (float, optional): size of the pixels in the y drection. If None, equals to pixel_size_x. Defaults to None.
        max_dx (int, optional): overlap in the x direction. Defaults to 0.
        max_dy (int, optional): overlap in the y direction. Defaults to 0.
        grid_width (int, optional): number of pixels in the width of one grid cell. Defaults to 256.
        grid_height (int, optional): number of pixels in the height of one grid cell. Defaults to 256.
        crs (str, optional): coordinate reference system. Defaults to 'EPSG:2056'.
        test_shape (shapely.geometry.base.BaseGeometry, optional): shape to test against for intersection. Defaults to None.

    Returns:
        GeoDataFrame: grid cells and their attributes
    """

    min_x, min_y = tile_origin

    number_cells_x, number_cells_y = get_grid_size(tile_size, grid_width, grid_height, max_dx, max_dy)

    # Convert dimensions from pixels to meters
    pixel_size_y = pixel_size_y if pixel_size_y else pixel_size_x
    grid_x_dim = grid_width * pixel_size_x
    grid_y_dim = grid_height * pixel_size_y
    max_dx_dim = max_dx * pixel_size_x
    max_dy_dim = max_dy * pixel_size_y

    # Create grid polygons
    polygons = []
    for x in range(number_cells_x):
        for y in range(number_cells_y):
            
            down_left = (min_x + x * (grid_x_dim - max_dx_dim), min_y + y * (grid_y_dim - max_dy_dim))

            # Fasten the process by not producing every single polygon
            if test_shape and not (test_shape.intersects(Point(down_left))):
                continue

            # Define the coordinates of the polygon vertices
            vertices = [down_left,
                        (min_x + (x + 1) * grid_x_dim - x * max_dx_dim, min_y + y * (grid_y_dim - max_dy_dim)),
                        (min_x + (x + 1) * grid_x_dim - x * max_dx_dim, min_y + (y + 1) * grid_y_dim - y * max_dy_dim),
                        (min_x + x * (grid_x_dim - max_dx_dim), min_y + (y + 1) * grid_y_dim - y * max_dy_dim)]

            # Create a Polygon object
            polygon = Polygon(vertices)
            polygons.append(polygon)

    # Create a GeoDataFrame from the polygons
    grid_gdf = GeoDataFrame(geometry=polygons, crs=crs)
    grid_gdf['id'] = [f'{round(min_x)}, {round(min_y)}' for min_x, min_y in [(poly.bounds[0], poly.bounds[1]) for poly in grid_gdf.geometry]]

    return grid_gdf


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

