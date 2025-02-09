import numpy as np
import pdb
import matplotlib.pyplot as plt
import random
from heapq import heappop, heappush
import time
import tracemalloc
import math

def parse_input_file(file_path):
    """
    Parse the input file to extract start-goal pairs and the grid.

    Args:
        file_path (str): Path to the input text file.

    Returns:
        tuple: A tuple containing:
            - List of start-goal pairs as tuples.
            - 2D list representing the grid.
    """
    start_goal_pairs = []
    grid = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    parsing_pairs = False
    parsing_grid = False

    for line in lines:
        line = line.strip()

        # Detect the start of start-goal pairs
        if line.startswith("# Start goal pairs:"):
            parsing_pairs = True
            continue

        # Parse start-goal pairs
        if parsing_pairs and line.startswith("#") and 'Start:' in line and 'Goal:' in line:
            start = tuple(map(int, line.split('Start:')[1].split('|')[0].strip()[1:-1].split(',')))
            goal = tuple(map(int, line.split('Goal:')[1].strip()[1:-1].split(',')))
            start_goal_pairs.append((start, goal))

        # Detect the start of the grid
        if not line.startswith("#") and ('.' in line or 'X' in line):
            parsing_pairs = False
            parsing_grid = True

        # Parse the grid
        if parsing_grid and ('.' in line or 'X' in line):
            # Parse the grid and convert to binary
            grid.append([1 if char == 'X' else 0 for char in line])


    return start_goal_pairs, grid

def bfs(grid, start, goal):
    """
    Perform Breadth-First Search (BFS) to find the shortest path between a start and goal cell
    in a grid with the origin (0, 0) at the top-left.

    In this version, coordinates are provided in (x, y) format, where:
      - x is the horizontal coordinate (column index)
      - y is the vertical coordinate (row index)
      
    The grid is a 2D list where:
      - 0 represents free space.
      - 1 represents an obstacle.
      
    Note: The grid is accessed as grid[y][x].

    Args:
        grid (list of lists): The 2D grid.
        start (tuple): Starting cell in (x, y) format.
        goal (tuple): Goal cell in (x, y) format.

    Returns:
        list: A list of (x, y) positions representing the path from start to goal (inclusive),
              or an empty list if no path exists.
    """
    from collections import deque

    # Define movements for a 4-connected grid.
    # With coordinates in (x, y):
    #   left: (-1, 0), right: (1, 0), up: (0, -1), down: (0, 1)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # The grid dimensions: number of rows and columns.
    rows = len(grid)         # vertical dimension (y)
    cols = len(grid[0])      # horizontal dimension (x)

    # Create a visited matrix for cells (indexed as visited[y][x]).
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    parent = {}  # Dictionary to reconstruct the path later.

    # Unpack the starting (x, y) coordinates.
    sx, sy = start
    # Mark the starting cell as visited.
    visited[sy][sx] = True

    # Initialize the queue with the start cell.
    queue = deque([start])

    while queue:
        current = queue.popleft()
        cx, cy = current

        # If the goal is reached, reconstruct the path by backtracking.
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent.get(current)
            return path[::-1]  # Return the reversed path.

        # Explore all 4-connected neighbors.
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            # Check if the new coordinates are within bounds.
            if 0 <= nx < cols and 0 <= ny < rows:
                # Only consider the neighbor if it hasn't been visited and is free (0).
                if not visited[ny][nx] and grid[ny][nx] == 0:
                    visited[ny][nx] = True
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

    return []  # Return an empty list if no path is found.

def dfs(grid, start, goal):
    """
    Perform Depth-First Search (DFS) to find a path from start to goal in a grid.

    The grid has its origin (0, 0) at the top-left, and coordinates are provided in (x, y) format:
      - x: column index
      - y: row index

    The grid is a 2D list where:
      - 0 represents free space.
      - 1 represents an obstacle.

    Args:
        grid (list of lists): The 2D grid.
        start (tuple): Starting position as (x, y).
        goal (tuple): Goal position as (x, y).

    Returns:
        list: A list of (x, y) positions representing the path from start to goal (inclusive),
              or an empty list if no path exists.
    """
    # Determine grid dimensions.
    rows = len(grid)
    cols = len(grid[0])
    
    # Initialize the stack, visited set, and parent dictionary.
    stack = [start]
    visited = {start}
    parent = {start: None}

    while stack:
        current = stack.pop()

        # If the goal is reached, reconstruct and return the path.
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]  # Reverse to get the path from start to goal.

        cx, cy = current  # current x and y.

        # Define neighbors in (x, y): left, right, up, down.
        neighbors = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
        for neighbor in neighbors:
            nx, ny = neighbor
            # Check if neighbor is within grid bounds.
            if 0 <= nx < cols and 0 <= ny < rows:
                # Proceed only if the neighbor cell is free (0) and hasn't been visited.
                if grid[ny][nx] == 0 and neighbor not in visited:
                    stack.append(neighbor)
                    visited.add(neighbor)  # Mark as visited immediately.
                    parent[neighbor] = current

    # If no path is found, return an empty list.
    return []

def heuristic(a, b):
    """
    Compute the Manhattan distance heuristic between two points.

    Coordinates are given in (x, y) format:
      - x: column index
      - y: row index

    Args:
        a (tuple): First point (x, y).
        b (tuple): Second point (x, y).

    Returns:
        int: Manhattan distance between a and b.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    """
    Perform A* search to find the shortest path from start to goal in a 4-connected grid.

    The grid has its origin (0, 0) at the top-left, and coordinates are provided in (x, y) format:
      - x: column index
      - y: row index

    The grid is a 2D list where:
      - 0 represents free space.
      - 1 represents an obstacle.
    
    Args:
        grid (list of lists): The 2D grid.
        start (tuple): Starting position as (x, y).
        goal (tuple): Goal position as (x, y).

    Returns:
        list: A list of positions (in (x, y) format) representing the path from start to goal (inclusive),
              or an empty list if no path exists.
    """
    # Define movements for a 4-connected grid: left, right, up, down.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Determine the grid dimensions.
    rows = len(grid)         # Number of rows (y-direction)
    cols = len(grid[0])      # Number of columns (x-direction)

    # Initialize the open set as a heap-based priority queue.
    open_set = []
    heappush(open_set, (0, start))  # Each item is a tuple (f_score, position).

    # Dictionaries to store the cost from start to each node and the estimated total cost.
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    # Dictionary to keep track of the parent of each node (for path reconstruction).
    parent = {}

    while open_set:
        # Pop the node with the lowest f_score.
        _, current = heappop(open_set)

        # If the goal is reached, reconstruct the path.
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent.get(current)
            return path[::-1]  # Reverse the path to go from start to goal.

        # Explore all 4-connected neighbors.
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            # Check if neighbor is within grid bounds:
            if 0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows:
                # Access the grid cell as grid[y][x]
                if grid[neighbor[1]][neighbor[0]] == 0:  # Check that the cell is free.
                    tentative_g_score = g_score[current] + 1  # Cost from start to neighbor.

                    # If this path to neighbor is better, record it.
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        parent[neighbor] = current
                        # To avoid duplicate entries, check if neighbor is already in open_set.
                        if neighbor not in [item[1] for item in open_set]:
                            heappush(open_set, (f_score[neighbor], neighbor))

    # If the goal was never reached, return an empty path.
    return []
def get_random_point(grid, goal, goal_bias=0.1):
    """
    Generate a random point in the grid with a bias towards the goal.
    
    Coordinates are in (x, y) format:
      - x: column index
      - y: row index
    """
    if random.random() < goal_bias:
        return goal
    rows, cols = len(grid), len(grid[0])
    while True:
        # Choose x from 0 to cols-1 and y from 0 to rows-1.
        x = random.randint(0, cols - 1)
        y = random.randint(0, rows - 1)
        if grid[y][x] == 0:  # Ensure the cell is free.
            return (x, y)

def euclidean_distance(p1, p2):
    """
    Compute the Euclidean distance between two points in (x, y) format.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))

def nearest_node(tree, point):
    """
    Find the nearest node in the tree to the given point using Euclidean distance.
    
    Both the tree keys and the point are in (x, y) format.
    """
    return min(tree.keys(), key=lambda node: euclidean_distance(node, point))

def steer(from_node, to_node, step_size=1):
    """
    Move from from_node towards to_node by a single step (4-connected movement).
    
    Coordinates are in (x, y) format.
    """
    dx, dy = to_node[0] - from_node[0], to_node[1] - from_node[1]
    if abs(dx) > abs(dy):
        # Move horizontally.
        new_point = (from_node[0] + (1 if dx > 0 else -1), from_node[1])
    else:
        # Move vertically.
        new_point = (from_node[0], from_node[1] + (1 if dy > 0 else -1))
    return new_point

def is_collision_free(grid, from_node, to_node):
    """
    Check if the path from from_node to to_node is collision-free.
    
    The function interpolates between the nodes in a straight line (using integer steps)
    and checks that all the cells along the way are free (i.e. grid[y][x] == 0).
    
    Coordinates are in (x, y) format.
    """
    x0, y0 = from_node
    x1, y1 = to_node
    dx, dy = x1 - x0, y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return grid[y0][x0] == 0
    rows, cols = len(grid), len(grid[0])
    for i in range(steps + 1):
        # Interpolate the coordinates.
        x = int(x0 + i * dx / steps)
        y = int(y0 + i * dy / steps)
        # Check bounds.
        if x < 0 or x >= cols or y < 0 or y >= rows:
            return False
        if grid[y][x] == 1:
            return False  # Collision detected.
    return True

def rrt(grid, start, goal, max_iterations=5000, step_size=1):
    """
    Perform RRT (Rapidly-exploring Random Tree) path planning.
    
    The grid uses (x, y) coordinates (x: column index, y: row index) with the top-left corner as (0, 0).
    Free space is represented by 0 and obstacles by 1.
    
    Args:
        grid (list of lists): The 2D grid.
        start (tuple): The start position as (x, y).
        goal (tuple): The goal position as (x, y).
        max_iterations (int): Maximum number of iterations to attempt.
        step_size (int): The step size for steering (typically 1 for grid movement).
        
    Returns:
        list: A list of positions (in (x, y) format) representing the path from start to goal, or an empty list if no path is found.
    """
    # The tree is a dictionary mapping a node to its parent.
    tree = {start: None}
    rows, cols = len(grid), len(grid[0])
    
    for _ in range(max_iterations):
        random_point = get_random_point(grid, goal)
        nearest = nearest_node(tree, random_point)
        new_point = steer(nearest, random_point, step_size)
        
        # Check that the new point is within the grid bounds.
        x, y = new_point
        if not (0 <= x < cols and 0 <= y < rows):
            continue
        
        if new_point not in tree and is_collision_free(grid, nearest, new_point):
            tree[new_point] = nearest

            # Optionally, you can check if the new point is close enough to the goal.
            # Here, we check for an exact match.
            if new_point == goal:
                path = []
                while new_point is not None:
                    path.append(new_point)
                    new_point = tree[new_point]
                return path[::-1]
                
    return []  # No path found.

def manhattan_distance(p1, p2):
    """
    Compute the Manhattan distance between two points in (x, y) format.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def plot_grid(grid, path=None, start=None, goal=None):
    # Create a color map for the grid
    cmap = plt.colormaps.get_cmap('Greys')
    cmap.set_under(color='white') # Free space color
    cmap.set_over(color='black') # Obstacle color
    grid_array = np.asarray(grid)
    fig, ax = plt.subplots()
    # Plot the grid with respect to the upper left-hand corner
    ax.matshow(grid_array, cmap=cmap, vmin=0.1, vmax=1.0, origin='upper')
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1))
    ax.set_yticks(np.arange(-0.5, len(grid), 1))
    ax.set_xticklabels(range(0, len(grid[0])+1))
    ax.set_yticklabels(range(0, len(grid)+1))
    # Plot the path with direction arrows
    if path:
        for i in range(len(path) - 1):
            start_y, start_x = path[i]
            end_y, end_x = path[i + 1]
            ax.arrow(start_y, start_x, end_y - start_y, end_x - start_x,
            head_width=0.3, head_length=0.3, fc='blue', ec='blue')
        # Plot the last point in the path
        ax.plot(path[-1][0], path[-1][1], 'b.')
    # Plot the start and goal points
    if start:
        ax.plot(start[0], start[1], 'go') # Start point in green
    if goal:
        ax.plot(goal[0], goal[1], 'ro') # Goal point in red

    print(f"Path: {path}")
    print("*"*60)    
    plt.show()
    return fig

def calculate_path_length(path):
    length = 0
    for i in range(1, len(path)):
        length += math.sqrt(sum((path[i][j] - path[i-1][j])**2 for j in range(len(path[i]))))
    return length

if __name__ == '__main__':
    file_path = 'Module_1_Path_Planning/maps/map2.txt'
    start_goal_pairs, grid = parse_input_file(file_path)

    for start, goal in start_goal_pairs:
        rows, cols = len(grid), len(grid[0])

        path = []

        # Check if start and goal positions are valid
        if not (0 <= start[0] < rows and 0 <= start[1] < cols and grid[start[1]][start[0]] == 0):
            print(f"Invalid start position: {start}. It is out of bounds or an obstacle.")
            fig = plot_grid(grid, path, start, goal)
            continue
        if not (0 <= goal[1] < rows and 0 <= goal[0] < cols and grid[goal[1]][goal[0]] == 0):
            print(f"Invalid goal position: {goal}. It is out of bounds or an obstacle.")
            fig = plot_grid(grid, path, start, goal)
            continue

        print("Start", grid[start[1]][start[0]])
        print("Goal", grid[goal[1]][goal[0]])

        print("Select pathfinding algorithm:")
        print("0: BFS")
        print("1: A*")
        print("2: RRT")
        print("3: DFS")
        choice = int(input("Enter choice: "))
        
        tracemalloc.start()
        start_time = time.time()
        
        if choice == 0:
            path = bfs(grid, start, goal)
        elif choice == 1:
            path = a_star(grid, start, goal)
        elif choice == 2:
            path = rrt(grid, start, goal)
        elif choice == 3:
            path = dfs(grid, start, goal)
        else:
            print("Invalid choice, defaulting to BFS")
            path = bfs(grid, start, goal)
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        execution_time = end_time - start_time
        path_length = calculate_path_length(path)
        
        print("*"*60)
        print(f"Execution Time: {execution_time} seconds")
        print(f"Current memory usage: {current / 10**6}MB; Peak: {peak / 10**6}MB")
        print(f"Path Length: {path_length}")
        print("*"*60)    

        fig = plot_grid(grid, path, start, goal)



  