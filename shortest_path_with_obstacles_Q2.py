def shortest_path_grid(grid, start, goal):
    
    #check if start is blocked
    if grid[start[0]][start[1]] == 1:
        return -1

    #check if goal is blocked
    if grid[goal[0]][goal[1]] == 1:
        return -1

    #vars to hold num of rows and columns
    rows = len(grid)
    columns = len(grid[0])
    
    #vars to hold start position
    sr = start[0]
    sc = start[1]

    #queues to track rows and columns
    rq = []
    cq = []

    #vars to track count and layers in search
    #count set to 1 to include initial position
    count = 1
    nodes_left = 1
    nodes_next = 0

    #flag to check if goal is reached
    final = False

    #populate matrix for visited flags
    #initial position flagged
    visited = [[False for x in range(columns)] for y in range(rows)]
    visited[sr][sc] = True

    #direction vectors for navigation
    dr = [-1, 1, 0, 0]
    dc = [0, 0, 1, -1]

    #add starting positions to respective queues
    rq.append(sr)
    cq.append(sc)
    
    #loop while queues are non empty
    while len(rq) > 0:

        #remove position coordinates from front of queue
        r = rq.pop(0)
        c = cq.pop(0)

        #check if goal is reached
        if (r,c) == goal:
            final = True
            break

        #loop through all possible neighboring spaces
        for i in range(4):
            rr = r + dr[i]
            cc = c + dc[i]

            #skip out of bounds, flagged and obstacle positions
            if rr < 0 or cc < 0:
                continue
            if rr >= rows or cc >= columns:
                continue
            if visited[rr][cc]:
                continue
            if grid[rr][cc] == 1:
                continue

            #add approved positions to queues
            #mark position visited
            rq.append(rr)
            cq.append(cc)
            visited[rr][cc] = True

            #increment next layer of nodes
            nodes_next = nodes_next + 1

        #verify when to increment count
        nodes_left = nodes_left - 1
        if nodes_left == 0:
            nodes_left = nodes_next
            nodes_next = 0
            count = count +1
            
    #check flag for completion return count
    if final:
        return count
    return -1


if __name__ == "__main__":
    grid = [[0,0,0],
            [1,1,0],
            [1,1,0]]
    start, goal = (0,1), (2,2)
    print(shortest_path_grid(grid, start, goal))
    assert shortest_path_grid(grid, start, goal) == 4

    grid = [[0,1],
            [1,0]]
    start, goal = (0, 0), (1,1)
    print(shortest_path_grid(grid, start, goal))
    assert shortest_path_grid(grid, start, goal) == -1

    #test case to see how well function works with no obstacles
    grid = [[0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]]
    start, goal = (0,0), (3,2)
    print(shortest_path_grid(grid, start, goal))
    assert shortest_path_grid(grid, start, goal) == 6

    #test case for when start position is blocked
    grid = [[1,0],
            [0,0]]
    start, goal = (0, 0), (1,1)
    print(shortest_path_grid(grid, start, goal))
    assert shortest_path_grid(grid, start, goal) == -1

    #test case for when goal position is blocked
    grid = [[0,0],
            [0,1]]
    start, goal = (0, 0), (1,1)
    print(shortest_path_grid(grid, start, goal))
    assert shortest_path_grid(grid, start, goal) == -1

    #additional test case from piazza
    grid = [[0,0,0,1,0,0,0,1,0,0],
            [0,1,0,1,0,1,0,1,0,0],
            [0,1,0,1,0,1,0,1,0,0],
            [0,1,0,1,0,1,0,1,0,0],
            [0,0,0,0,0,1,0,0,0,0]]
    start, goal = (2,0), (1,8)
    print(shortest_path_grid(grid, start, goal))
    assert shortest_path_grid(grid, start, goal) == 22

