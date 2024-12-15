#%% md
# ### 1. Imports and Constants
#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math
from shapely.geometry import Point, Polygon, box
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from scipy.interpolate import griddata
import matplotlib as mpl
import plotly.graph_objects as go
from flask import Flask, render_template_string, request

mpl.rcParams['figure.dpi'] = 400

# Constants for map generation
MAP_WIDTH = 100
MAP_HEIGHT = 100
WATER = 'water'
LAND = 'land'
GRID_UNIT_NAUTICAL_MILES = 1

# Ship parameters
mass = 50000  # kg
base_velocity = 15 * 0.51444  # Convert knots to m/s (15 knots)
hull_length = 50  # meters
frontal_area = 100  # m²
water_density = 1025  # kg/m³
efficiency = 0.4
fuel_energy_density = 42e6  # J/kg

# Weather impact factors
WEATHER_SPEED_FACTOR = {
    'calm': 1.0,
    'windy': 0.9,
    'storm': 0.7
}
WEATHER_EMISSION_FACTOR = {
    'calm': 1.0,
    'windy': 1.1,
    'storm': 1.3
}
WIND_SPEED_RANGES = {
    'calm': (0, 5),
    'windy': (5, 15),
    'storm': (15, 30)
}

# Geographical bounds for Caspian Sea region
LAT_MIN, LAT_MAX = 36.0, 44.0
LON_MIN, LON_MAX = 48.0, 54.0

app = Flask(__name__)

#%% md
# ### 2. Data Loading and Map Initialization
#%%
# Path to the downloaded shapefile
shapefile_path = 'ne_10m_land/ne_10m_land.shp'
world = gpd.read_file(shapefile_path)
world = world.to_crs("EPSG:4326")

# Focus on the region of interest
region_of_interest = world.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX]

# Create grid for the region
lat_range = LAT_MAX - LAT_MIN
lon_range = LON_MAX - LON_MIN
lat_per_cell = lat_range / MAP_HEIGHT
lon_per_cell = lon_range / MAP_WIDTH

lats = np.linspace(LAT_MIN, LAT_MAX, MAP_HEIGHT)
lons = np.linspace(LON_MIN, LON_MAX, MAP_WIDTH)

# GeoDataFrame for grid cells
grid_cells = []
for i in range(MAP_HEIGHT):
    for j in range(MAP_WIDTH):
        cell_box = box(
            LON_MIN + j * lon_per_cell,
            LAT_MIN + i * lat_per_cell,
            LON_MIN + (j + 1) * lon_per_cell,
            LAT_MIN + (i + 1) * lat_per_cell
        )
        grid_cells.append({'geometry': cell_box, 'row': i, 'col': j})
grid_gdf = gpd.GeoDataFrame(grid_cells, crs="EPSG:4326")

# Determine land or water
land_union = region_of_interest.unary_union
grid_gdf['land'] = grid_gdf['geometry'].intersects(land_union)

# Map data array
map_data = np.full((MAP_HEIGHT, MAP_WIDTH), WATER, dtype=object)
land_indices = grid_gdf[grid_gdf['land'] == True][['row', 'col']].values
map_data[land_indices[:, 0], land_indices[:, 1]] = LAND

#%% md
# ### 3. Map and Weather Data Generation
#%%
# Depth parameters
MIN_DEPTH = 10  # Minimum depth in meters
MAX_DEPTH = 1000  # Maximum depth in meters


def generate_depth_map(map_data):
    depth_map = np.zeros(map_data.shape)
    water_mask = map_data == WATER
    depth_map[water_mask] = np.random.uniform(MIN_DEPTH, MAX_DEPTH, size=water_mask.sum())
    return depth_map


# def assign_weather_and_wind(map_data):
#     weather_conditions = ['calm', 'windy', 'storm']
#     water_mask = map_data == WATER
#     weather_map = np.empty(map_data.shape, dtype=object)
#     wind_map = np.empty(map_data.shape, dtype=object)

#     num_water_cells = water_mask.sum()
#     weather_choices = np.random.choice(weather_conditions, size=num_water_cells, p=[0.6, 0.35, 0.05])
#     weather_map[water_mask] = weather_choices

#     for weather_condition in weather_conditions:
#         indices = np.where((weather_map == weather_condition) & water_mask)
#         wind_speeds = np.random.uniform(*WIND_SPEED_RANGES[weather_condition], size=len(indices[0]))
#         wind_directions = np.random.uniform(0, 360, size=len(indices[0]))
#         for idx in range(len(indices[0])):
#             i, j = indices[0][idx], indices[1][idx]
#             wind_map[i, j] = {'speed': wind_speeds[idx], 'direction': wind_directions[idx]}

#     weather_map[~water_mask] = 'land'
#     wind_map[~water_mask] = None

#     return weather_map, wind_map

def assign_weather_and_wind(map_data):
    weather_conditions = ['calm', 'windy', 'storm']
    water_mask = map_data == WATER
    weather_map = np.empty(map_data.shape, dtype=object)
    wind_map = np.empty(map_data.shape, dtype=object)

    num_water_cells = water_mask.sum()
    weather_choices = np.random.choice(weather_conditions, size=num_water_cells, p=[0.6, 0.35, 0.05])
    weather_map[water_mask] = weather_choices

    # Frequency factors for smoother wind patterns
    freq_x, freq_y = 0.1, 0.1  # These control how "wavy" the wind pattern is
    noise_level = 0.1  # Controls how much noise to add to the wind

    # Generate smooth wind patterns over the map coordinates
    grid_x, grid_y = np.meshgrid(np.arange(map_data.shape[1]), np.arange(map_data.shape[0]))

    for weather_condition in weather_conditions:
        indices = np.where((weather_map == weather_condition) & water_mask)

        # Define base wind speeds for each condition (sin/cos waves)
        base_wind_speeds = (np.sin(grid_x * freq_x) + np.cos(grid_y * freq_y)) / 2.0
        base_wind_speeds = base_wind_speeds[indices]  # Extract only relevant points
        wind_speeds = base_wind_speeds * WIND_SPEED_RANGES[weather_condition][1]  # Scale based on max speed

        # Generate smooth wind directions using sin/cos
        base_wind_directions = (np.sin(grid_x * freq_x) + np.cos(grid_y * freq_y)) * 180.0  # Map to degrees
        base_wind_directions = base_wind_directions[indices]  # Extract relevant points
        wind_directions = base_wind_directions % 360  # Keep in 0-360 range

        # Add small random noise to speed and direction
        wind_speeds += np.random.uniform(-noise_level / 2, noise_level / 2, size=len(indices[0]))
        wind_directions += np.random.uniform(-noise_level * 180, noise_level * 180, size=len(indices[0]))

        # Assign wind conditions to the map
        for idx in range(len(indices[0])):
            i, j = indices[0][idx], indices[1][idx]
            wind_map[i, j] = {'speed': max(0, wind_speeds[idx]), 'direction': wind_directions[idx]}

    weather_map[~water_mask] = 'land'
    wind_map[~water_mask] = None

    return weather_map, wind_map


#%% md
# ### 4. Helper Functions for Calculations
#%%
def haversine(coord1, coord2):
    i1, j1 = coord1
    i2, j2 = coord2
    lat1 = LAT_MIN + i1 * lat_per_cell + lat_per_cell / 2
    lon1 = LON_MIN + j1 * lon_per_cell + lon_per_cell / 2
    lat2 = LAT_MIN + i2 * lat_per_cell + lat_per_cell / 2
    lon2 = LON_MIN + j2 * lon_per_cell + lon_per_cell / 2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    R = 6371000  # Radius of Earth in meters
    distance = R * c
    return distance


# Fuel consumption and drag calculations
def fuel_consumption_turning_with_drag(mass, velocity, distance, rudder_angle_deg, hull_length, frontal_area,
                                       water_density, efficiency, fuel_energy_density):
    g = 9.81  # Gravity, m/s^2
    kinematic_viscosity_water = 1e-6  # Kinematic viscosity of water, m^2/s
    rudder_angle_rad = math.radians(rudder_angle_deg)  # Convert rudder angle to radians

    # Calculate radius of turn
    if rudder_angle_rad == 0:
        radius_of_turn = float('inf')
    else:
        radius_of_turn = velocity ** 2 / (g * math.tan(rudder_angle_rad))

    reynolds_number = (velocity * hull_length) / kinematic_viscosity_water  # Reynolds number
    frictional_resistance_coefficient = 0.075 / (
                math.log10(reynolds_number) - 2) ** 2  # Frictional resistance coefficient using ITTC-1957 formula
    form_factor = 0.2  # Form factor (k) for the hull shape. Modify this value based on ship's hull type if known
    drag_coefficient = (1 + form_factor) * frictional_resistance_coefficient  # Total drag coefficient
    drag_force = 0.5 * drag_coefficient * water_density * velocity ** 2 * frontal_area  # Calculate drag force
    friction_force = 0.01 * mass * g  # Assume friction coefficient ~ 0.01 for water

    # Centripetal force
    if radius_of_turn != float('inf'):
        centripetal_force = (mass * velocity ** 2) / radius_of_turn
    else:
        centripetal_force = 0

    total_force = drag_force + friction_force + centripetal_force
    power_required = total_force * velocity
    time = distance / velocity
    energy_required = power_required * time
    energy_required = energy_required / efficiency
    fuel_consumption = energy_required / fuel_energy_density

    return {
        "fuel_consumption": fuel_consumption,
        "radius_of_turn": radius_of_turn,
        "drag_coefficient": drag_coefficient,
        "drag_force": drag_force,
        "time": time
    }


#%% md
# ### 5. Pathfinding Algorithms
#%%
def astar(map_data, start, goal, cost_func, heuristic_func, weather_map=None, depth_map=None, wind_map=None):
    # Directions include diagonals
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic_func(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1], gscore[goal]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if 0 <= neighbor[0] < map_data.shape[0]:
                if 0 <= neighbor[1] < map_data.shape[1]:
                    if map_data[neighbor[0]][neighbor[1]] != WATER:
                        continue
                else:
                    continue
            else:
                continue

            tentative_g_score = gscore[current] + cost_func(current, neighbor, weather_map, depth_map, wind_map)

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic_func(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False, None


# Cost functions
def uniform_cost(current, neighbor, weather_map=None, depth_map=None, wind_map=None):
    # Distance between current and neighbor
    distance = haversine(current, neighbor)
    return distance  # Uniform cost per unit distance


def movement_cost(current, neighbor, weather_map, depth_map, wind_map):
    # Distance between current and neighbor in meters
    distance = haversine(current, neighbor)

    # Weather and depth factors
    weather = weather_map[neighbor[0]][neighbor[1]]
    depth = depth_map[neighbor[0]][neighbor[1]]

    # Get wind information
    wind = wind_map[neighbor[0]][neighbor[1]]
    if wind is None:
        wind_speed = 0
        wind_direction = 0
    else:
        wind_speed = wind['speed'] * 0.51444  # Convert knots to m/s
        wind_direction = wind['direction']

    # Calculate ship movement direction
    delta_i = neighbor[0] - current[0]
    delta_j = neighbor[1] - current[1]
    movement_direction = (math.degrees(math.atan2(-delta_i, delta_j)) + 360) % 360

    # Wind components
    wx = wind_speed * math.cos(math.radians(wind_direction))
    wy = wind_speed * math.sin(math.radians(wind_direction))

    # Adjust velocity based on weather conditions
    speed_factor = WEATHER_SPEED_FACTOR.get(weather, 1.0)
    v_ship = base_velocity * speed_factor

    v_ship_x = v_ship * math.cos(math.radians(movement_direction))
    v_ship_y = v_ship * math.sin(math.radians(movement_direction))

    v_eff_x = v_ship_x + wx
    v_eff_y = v_ship_y + wy

    v_eff = math.sqrt(v_eff_x ** 2 + v_eff_y ** 2)
    v_eff = max(v_eff, 0.1)  # Prevent division by zero

    # Compute rudder angle (assumed zero for simplicity)
    rudder_angle = 0

    # Compute fuel consumption using detailed physics
    result = fuel_consumption_turning_with_drag(
        mass, v_eff, distance, rudder_angle, hull_length, frontal_area,
        water_density, efficiency, fuel_energy_density)

    fuel_consumption = result['fuel_consumption']

    # Cost can be fuel consumption or time; here we use fuel consumption as cost
    cost = fuel_consumption
    return cost


# Heuristic functions
def heuristic(a, b):
    # Haversine distance
    return haversine(a, b)


def heuristic_cost_estimate(a, b):
    # Simplified heuristic based on minimum possible cost
    distance = haversine(a, b)
    MIN_VELOCITY = base_velocity * max(WEATHER_SPEED_FACTOR.values())
    MIN_VELOCITY = max(MIN_VELOCITY, 0.1)
    MIN_TIME = distance / MIN_VELOCITY
    MIN_POWER = 0  # Assume minimal power required
    MIN_ENERGY = MIN_POWER * MIN_TIME
    MIN_FUEL_CONSUMPTION = MIN_ENERGY / fuel_energy_density / efficiency
    heuristic_cost = MIN_FUEL_CONSUMPTION
    return heuristic_cost


#%% md
# ### 6. Comparison
#%%
# Calculate cost and emissions, including total distance
def calculate_route_metrics(path, weather_map, depth_map, wind_map):
    total_time = 0
    total_fuel = 0
    total_distance = 0
    for idx in range(len(path) - 1):
        current = path[idx]
        next_node = path[idx + 1]
        # Distance between current and next_node in meters
        distance = haversine(current, next_node)
        total_distance += distance

        # Weather and depth factors
        weather = weather_map[next_node[0]][next_node[1]]
        depth = depth_map[next_node[0]][next_node[1]]

        # Get wind information
        wind = wind_map[next_node[0]][next_node[1]]
        if wind is None:
            wind_speed = 0
            wind_direction = 0
        else:
            wind_speed = wind['speed'] * 0.51444  # Convert knots to m/s
            wind_direction = wind['direction']

        # Calculate ship movement direction
        delta_i = next_node[0] - current[0]
        delta_j = next_node[1] - current[1]
        movement_direction = (math.degrees(math.atan2(-delta_i, delta_j)) + 360) % 360

        # Wind components
        wx = wind_speed * math.cos(math.radians(wind_direction))
        wy = wind_speed * math.sin(math.radians(wind_direction))

        # Adjust velocity based on weather conditions
        speed_factor = WEATHER_SPEED_FACTOR.get(weather, 1.0)
        v_ship = base_velocity * speed_factor

        v_ship_x = v_ship * math.cos(math.radians(movement_direction))
        v_ship_y = v_ship * math.sin(math.radians(movement_direction))

        v_eff_x = v_ship_x + wx
        v_eff_y = v_ship_y + wy

        v_eff = math.sqrt(v_eff_x ** 2 + v_eff_y ** 2)
        v_eff = max(v_eff, 0.1)  # Prevent division by zero

        # Compute rudder angle (assumed zero for simplicity)
        rudder_angle = 0

        # Compute fuel consumption using detailed physics
        result = fuel_consumption_turning_with_drag(
            mass, v_eff, distance, rudder_angle, hull_length, frontal_area,
            water_density, efficiency, fuel_energy_density)

        fuel_consumption = result['fuel_consumption']
        time = result['time']

        total_time += time
        total_fuel += fuel_consumption

    total_distance /= 1852  # Convert meters to nautical miles

    metrics = {
        'distance (nautical miles)': total_distance,
        'time (hours)': total_time / 3600,
        'fuel (kg)': total_fuel
    }
    return metrics


#%% md
# ### 7. Select origin and destination
#%%
def is_border_cell(map_data, cell):
    x, y = cell
    # Check if any neighboring cell is land (4-neighborhood)
    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    for nx, ny in neighbors:
        if 0 <= nx < map_data.shape[0] and 0 <= ny < map_data.shape[1]:  # Ensure within bounds
            if map_data[nx, ny] == LAND:
                return True
    return False


def find_closest_cell(coordinates, cells):
    closest_cell = None
    min_distance = float('inf')

    for cell in cells:
        # Calculate Euclidean distance
        distance = np.linalg.norm(np.array(cell) - np.array(coordinates))

        # Check if this is the closest cell
        if distance < min_distance:
            min_distance = distance
            closest_cell = cell

    return closest_cell


def select_border_water_points(map_data):
    # Find water cells that are adjacent to land cells
    border_water_cells = [cell for cell in np.argwhere(map_data == WATER) if is_border_cell(map_data, cell)]

    if len(border_water_cells) < 2:
        raise ValueError("Not enough border water cells to select start and goal.")

    # Randomly select two border water cells far apart
    start = border_water_cells[np.random.choice(len(border_water_cells))]
    goal = border_water_cells[np.random.choice(len(border_water_cells))]

    # Ensure they are far apart
    while np.linalg.norm(start - goal) < max(MAP_WIDTH, MAP_HEIGHT) / 2:
        goal = border_water_cells[np.random.choice(len(border_water_cells))]

    return tuple(start), tuple(goal)


#%% md
# ### 8. Visualization
#%%
def plot_map(map_data, start, goal, path=None, optimized_path=None, weather_map=None, wind_map=None):
    # Define the projection
    projection = ccrs.PlateCarree()

    # Create a figure with Cartopy and set a black background
    fig, ax = plt.subplots(figsize=(16, 12), subplot_kw={'projection': projection}, facecolor='black')
    ax.set_facecolor('black')  # Set the axes background to black

    # Set the extent to the Caspian Sea region
    ax.set_extent([LON_MIN - 1, LON_MAX + 1, LAT_MIN - 1, LAT_MAX + 1], crs=projection)

    # Add map features with colors suited for a black background
    ax.add_feature(cfeature.LAND, facecolor='dimgray')
    ax.add_feature(cfeature.OCEAN, facecolor='black')
    ax.add_feature(cfeature.COASTLINE, edgecolor='white')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='lightgray')
    ax.add_feature(cfeature.RIVERS, edgecolor='lightblue')
    ax.add_feature(cfeature.LAKES, edgecolor='lightblue', facecolor='black', alpha=0.7)

    # Overlay the synthetic map using pcolormesh with adjusted colors
    land_binary = np.where(map_data == LAND, 1, 0)
    lat_grid = np.linspace(LAT_MIN, LAT_MAX, MAP_HEIGHT + 1)
    lon_grid = np.linspace(LON_MIN, LON_MAX, MAP_WIDTH + 1)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)

    if weather_map is not None and wind_map is not None:
        # Create a 2D array for wind speed (m/s)
        wind_speed_map = np.full(map_data.shape, np.nan)
        for i in range(MAP_HEIGHT):
            for j in range(MAP_WIDTH):
                if map_data[i][j] == WATER and wind_map[i][j] is not None:
                    wind_speed_map[i, j] = wind_map[i][j]['speed'] * 0.51444  # Convert knots to m/s

        # Apply the mask to remove land data BEFORE interpolation
        wind_speed_map_masked = np.ma.masked_where(map_data == LAND, wind_speed_map)

        # Create meshgrid for current grid (100x100)
        x = np.linspace(LON_MIN, LON_MAX, MAP_WIDTH)
        y = np.linspace(LAT_MIN, LAT_MAX, MAP_HEIGHT)
        X, Y = np.meshgrid(x, y)

        # Mask invalid (NaN) wind speed data
        mask = ~np.isnan(wind_speed_map_masked)

        # Perform interpolation for smoothness
        # Define a finer grid for interpolation (500x500)
        fine_lon = np.linspace(LON_MIN, LON_MAX, MAP_WIDTH * 5)
        fine_lat = np.linspace(LAT_MIN, LAT_MAX, MAP_HEIGHT * 5)
        Fine_X, Fine_Y = np.meshgrid(fine_lon, fine_lat)

        # Interpolate using cubic method on the masked wind speed map
        wind_speed_fine = griddata(
            points=(X[mask], Y[mask]),
            values=wind_speed_map_masked[mask],
            xi=(Fine_X, Fine_Y),
            method='cubic'
        )

        # No need to mask again here, since the mask was applied before interpolation

        # Plot the wind speed heatmap using contourf
        contour = ax.contourf(
            Fine_X, Fine_Y, wind_speed_fine,
            levels=100, cmap='Blues', alpha=0.6,  # Adjust alpha for transparency
            transform=projection
        )

        # Add a color bar for wind speed
        cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, aspect=50)
        cbar.set_label('Wind Speed (m/s)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.tick_params(labelcolor='white')

    # Plot start and goal points with colors that stand out on black
    start_lon = LON_MIN + (start[1] + 0.5) * lon_per_cell
    start_lat = LAT_MIN + (start[0] + 0.5) * lat_per_cell
    goal_lon = LON_MIN + (goal[1] + 0.5) * lon_per_cell
    goal_lat = LAT_MIN + (goal[0] + 0.5) * lat_per_cell

    ax.scatter(start_lon, start_lat, color='cyan', edgecolors='white', s=100, label='Start', transform=projection,
               zorder=5)
    ax.scatter(goal_lon, goal_lat, color='magenta', edgecolors='white', s=100, label='Goal', transform=projection,
               zorder=5)

    # Plot land
    ax.add_feature(cfeature.LAND, facecolor='dimgrey', alpha=0.9, zorder=1)

    # Plot paths if they exist
    # Line outline
    if path:
        path_lons = [LON_MIN + (j + 0.5) * lon_per_cell for (_, j) in path]
        path_lats = [LAT_MIN + (i + 0.5) * lat_per_cell for (i, _) in path]
        ax.plot(path_lons, path_lats, color='white', linewidth=3, transform=projection)

    # Acutal line
    if path:
        path_lons = [LON_MIN + (j + 0.5) * lon_per_cell for (_, j) in path]
        path_lats = [LAT_MIN + (i + 0.5) * lat_per_cell for (i, _) in path]
        ax.plot(path_lons, path_lats, color='red', linewidth=2, label='Shortest Path', transform=projection)

    # Line outline
    if optimized_path:
        opt_path_lons = [LON_MIN + (j + 0.5) * lon_per_cell for (_, j) in optimized_path]
        opt_path_lats = [LAT_MIN + (i + 0.5) * lat_per_cell for (i, _) in optimized_path]
        ax.plot(opt_path_lons, opt_path_lats, color='white', linewidth=3, linestyle='-', transform=projection)

    # Actual line
    if optimized_path:
        opt_path_lons = [LON_MIN + (j + 0.5) * lon_per_cell for (_, j) in optimized_path]
        opt_path_lats = [LAT_MIN + (i + 0.5) * lat_per_cell for (i, _) in optimized_path]
        ax.plot(opt_path_lons, opt_path_lats, color='lime', linewidth=2, linestyle='-', label='Optimized Path',
                transform=projection)

    # Modify gridlines to be visible on a black background
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='gray', linestyle='--',
                      linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'color': 'white'}
    gl.ylabel_style = {'color': 'white'}

    # Add legend and title with colors suited for a black background
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
    plt.title('Maritime Route Optimization near the Caspian Sea, Azerbaijan', color='white', fontsize=16)
    plt.show()


def plot_map_plotly(map_data, start, goal, path=None, optimized_path=None, weather_map=None, wind_map=None):
    # Define the mapbox token (you need to have your own token from mapbox.com)
    mapbox_token = 'pk.eyJ1IjoiaW1yYW5qYWJyYXlpbG92IiwiYSI6ImNtMjc5ZDIzMjE3eHoycXFzdXpibDJycG8ifQ.n95rD4KbWRdtkmPRR0pguQ'

    # Create base figure
    fig = go.Figure()

    # Add water and land (assuming map_data is binary for simplicity)
    land_binary = np.where(map_data == LAND, 1, 0)

    # Plot the base map (You can use scattermapbox to show points)
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[LON_MIN, LON_MAX],
        lat=[LAT_MIN, LAT_MAX],
        marker=dict(size=0),  # Invisible to just set the extent
        showlegend=False
    ))

    # Add start and goal points
    start_lon = LON_MIN + (start[1] + 0.5) * lon_per_cell
    start_lat = LAT_MIN + (start[0] + 0.5) * lat_per_cell
    goal_lon = LON_MIN + (goal[1] + 0.5) * lon_per_cell
    goal_lat = LAT_MIN + (goal[0] + 0.5) * lat_per_cell

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[start_lon, goal_lon],
        lat=[start_lat, goal_lat],
        marker=dict(size=14, color=['cyan', 'magenta'], opacity=0.8),
        text=["Start", "Goal"],
        name="Start/Goal"
    ))

    # Plot paths if they exist
    if path:
        path_lons = [LON_MIN + (j + 0.5) * lon_per_cell for (_, j) in path]
        path_lats = [LAT_MIN + (i + 0.5) * lat_per_cell for (i, _) in path]

        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=path_lons,
            lat=path_lats,
            line=dict(width=4, color='red'),
            name="Shortest Path"
        ))

    if optimized_path:
        opt_path_lons = [LON_MIN + (j + 0.5) * lon_per_cell for (_, j) in optimized_path]
        opt_path_lats = [LAT_MIN + (i + 0.5) * lat_per_cell for (i, _) in optimized_path]

        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=opt_path_lons,
            lat=opt_path_lats,
            line=dict(width=4, color='lime'),
            name="Optimized Path"
        ))

    # Add wind map as contour if available
    if wind_map is not None:
        wind_speed_map = np.full(map_data.shape, np.nan)
        for i in range(MAP_HEIGHT):
            for j in range(MAP_WIDTH):
                if map_data[i][j] == WATER and wind_map[i][j] is not None:
                    wind_speed_map[i, j] = wind_map[i][j]['speed'] * 0.51444  # Convert knots to m/s

        # Flatten data for contour plot
        wind_speeds_flat = wind_speed_map.flatten()
        lons_flat = np.linspace(LON_MIN, LON_MAX, MAP_WIDTH).repeat(MAP_HEIGHT)
        lats_flat = np.tile(np.linspace(LAT_MIN, LAT_MAX, MAP_HEIGHT), MAP_WIDTH)

        fig.add_trace(go.Densitymapbox(
            lon=lons_flat,
            lat=lats_flat,
            z=wind_speeds_flat,
            radius=10,
            colorscale='Blues',
            opacity=0.6,
            name="Wind Speed (m/s)"
        ))

    # Update layout for mapbox
    fig.update_layout(
        mapbox=dict(
            accesstoken=mapbox_token,
            style="dark",  # You can change the style to suit your needs
            center=dict(lon=(LON_MIN + LON_MAX) / 2, lat=(LAT_MIN + LAT_MAX) / 2),
            zoom=6
        ),
        showlegend=True,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    # Return the figure as HTML
    return fig.to_html()


def lat_lon_to_grid_cell(lat, lon):
    """
    Converts latitude and longitude to the nearest grid cell (i, j) in map_data.
    """
    if lat < LAT_MIN or lat > LAT_MAX or lon < LON_MIN or lon > LON_MAX:
        raise ValueError("Coordinates out of bounds")

    i = int((lat - LAT_MIN) / lat_per_cell)
    j = int((lon - LON_MIN) / lon_per_cell)

    # Clamp to the grid boundaries
    i = min(max(i, 0), MAP_HEIGHT - 1)
    j = min(max(j, 0), MAP_WIDTH - 1)

    return i, j


#%% md
# ### 9. Endpoints
#%%
@app.route('/')
def index():
    # Extract start and goal coordinates from query parameters (latitude and longitude)
    start_lat = float(request.args.get('start_lat', 0.0))
    start_lon = float(request.args.get('start_lon', 0.0))
    goal_lat = float(request.args.get('goal_lat', 0.0))
    goal_lon = float(request.args.get('goal_lon', 0.0))

    # Generate map_data
    weather_map, wind_map = assign_weather_and_wind(map_data)
    depth_map = generate_depth_map(map_data)

    # Map latitude/longitude to grid cells
    if start_lat == 0.0 or start_lon == 0.0 or goal_lat == 0.0 or goal_lon == 0.0:
        start, goal = select_border_water_points(map_data)
    else:
        try:
            start = lat_lon_to_grid_cell(start_lat, start_lon)
            goal = lat_lon_to_grid_cell(goal_lat, goal_lon)
        except ValueError as e:
            return str(e), 400

    # Run A* to calculate the initial and optimized paths
    initial_path, _ = astar(map_data, start, goal, uniform_cost, heuristic)
    optimized_path, _ = astar(map_data, start, goal, movement_cost, heuristic, weather_map, depth_map, wind_map)

    # Calculate route metrics if paths are found
    initial_metrics = calculate_route_metrics(initial_path, weather_map, depth_map, wind_map) if initial_path else None
    optimized_metrics = calculate_route_metrics(optimized_path, weather_map, depth_map,
                                                wind_map) if optimized_path else None

    if initial_metrics is not None and initial_metrics['fuel (kg)'] < optimized_metrics['fuel (kg)']:
        initial_metrics, optimized_metrics = optimized_metrics, initial_metrics

    # Plot the map with Plotly and return the plot as HTML
    plot_html = plot_map_plotly(map_data, start, goal, initial_path, optimized_path, weather_map, wind_map)

    # Render the HTML page with the plot
    return render_template_string("""
    <html>
        <head>
            <title>Plotly Map</title>
            <style>
                body {
                    background-color: black;
                }
                /* Make the container relative so that metrics can be positioned on top */
                .map-container {
                    position: relative;
                    width: 100%;
                    height: 100%;
                }

                /* Style the floating metrics box */
                .metrics-box {
                    position: absolute;
                    top: 20px;
                    left: 20px;
                    background-color: rgba(255, 255, 255, 0.8);
                    padding: 10px;
                    border-radius: 8px;
                    z-index: 1000;
                    width: 300px;
                }
            </style>
        </head>
        <body>
            <h1>Maritime Route Optimization</h1>

            <div class="map-container">
                <div class="metrics-box">
                    <h2>Initial Route Metrics</h2>
                    {% if initial_metrics %}
                        <ul>
                            {% for key, value in initial_metrics.items() %}
                                <li><strong>{{ key }}:</strong> {{ "%.2f"|format(value) }}</li>
                            {% endfor %}
                        </ul>
                    {% else %}
                        <p>No initial route found.</p>
                    {% endif %}

                    <h2>Optimized Route Metrics</h2>
                    {% if optimized_metrics %}
                        <ul>
                            {% for key, value in optimized_metrics.items() %}
                                <li><strong>{{ key }}:</strong> {{ "%.2f"|format(value) }}</li>
                            {% endfor %}
                        </ul>
                    {% else %}
                        <p>No optimized route found.</p>
                    {% endif %}
                </div>

                {{ plot_html | safe }}
            </div>
        </body>
    </html>
    """, plot_html=plot_html, initial_metrics=initial_metrics, optimized_metrics=optimized_metrics)


#%% md
# ### 10. Main
#%%
def main():
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
#%%
