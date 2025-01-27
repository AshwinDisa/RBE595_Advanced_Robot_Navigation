import numpy as np
import pdb
import matplotlib.pyplot as plt

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
    Perform BFS to find the shortest path from start to goal in a 4-connected grid.

    Args:
        grid (list): 2D list representing the binary grid (0 for free space, 1 for obstacles).
        start (tuple): Starting position (row, col).
        goal (tuple): Goal position (row, col).

    Returns:
        list: List of positions representing the path from start to goal, or an empty list if no path exists.
    """
    from collections import deque

    # Directions for 4-connected movement (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    parent = {}  # To reconstruct the path

    queue = deque([start])
    visited[start[0]][start[1]] = True

    while queue:
        current = queue.popleft()

        # If goal is reached, reconstruct the path
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent.get(current)
            return path[::-1]  # Reverse the path

        # Explore neighbors
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:  # Check bounds
                if not visited[neighbor[0]][neighbor[1]] and grid[neighbor[0]][neighbor[1]] == 0:  # Check free space
                    visited[neighbor[0]][neighbor[1]] = True
                    parent[neighbor] = current
                    queue.append(neighbor)

    return []  # No path found

def heuristic(a, b):
    """
    Compute the Manhattan distance heuristic between two points.

    Args:
        a (tuple): First point (row, col).
        b (tuple): Second point (row, col).

    Returns:
        int: Manhattan distance between a and b.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    """
    Perform A* search to find the shortest path from start to goal in a 4-connected grid.

    Args:
        grid (list): 2D list representing the binary grid (0 for free space, 1 for obstacles).
        start (tuple): Starting position (row, col).
        goal (tuple): Goal position (row, col).

    Returns:
        list: List of positions representing the path from start to goal, or an empty list if no path exists.
    """
    from heapq import heappop, heappush

    # Directions for 4-connected movement (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    rows, cols = len(grid), len(grid[0])
    open_set = []
    heappush(open_set, (0, start))  # (priority, position)

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    parent = {}  # To reconstruct the path

    while open_set:
        _, current = heappop(open_set)

        # If goal is reached, reconstruct the path
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent.get(current)
            return path[::-1]  # Reverse the path

        # Explore neighbors
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:  # Check bounds
                if grid[neighbor[0]][neighbor[1]] == 0:  # Check free space
                    tentative_g_score = g_score[current] + 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        parent[neighbor] = current

                        if neighbor not in [item[1] for item in open_set]:
                            heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found

def plot_grid(grid, path=None, start=None, goal=None):
    # Create a color map for the grid
    cmap = plt.cm.get_cmap('Greys')
    cmap.set_under(color='white') # Free space color
    cmap.set_over(color='black') # Obstacle color
    grid_array = np.asarray(grid)
    fig, ax = plt.subplots()
    # Plot the grid with respect to the upper left-hand corner
    ax.matshow(grid_array, cmap=cmap, vmin=0.1, vmax=1.0, origin='lower')
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
            ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
            head_width=0.3, head_length=0.3, fc='blue', ec='blue')
        # Plot the last point in the path
        ax.plot(path[-1][0], path[-1][1], 'b.')
    # Plot the start and goal points
    if start:
        ax.plot(start[1], start[0], 'go') # Start point in green
    if goal:
        ax.plot(goal[1], goal[0], 'ro') # Goal point in red

    print(f"Path {i+1}: {path}")
    plt.show()
    return fig  

if __name__ == '__main__':
    
    file_path = 'Module_1_Path_Planning/maps/map1.txt'
    start_goal_pairs, grid = parse_input_file(file_path)
    for i in range(len(start_goal_pairs)):
        start, goal = start_goal_pairs[i]
        # path = bfs(grid, start, goal)
        path = a_star(grid, start, goal)
        fig = plot_grid(grid, path, start, goal)


  