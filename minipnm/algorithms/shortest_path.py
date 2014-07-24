class PathNotFound(Exception):
    msg = "{} options exhausted, path not found"
    def __init__(self, exhausted):
        self.exhausted = list(exhausted)
    def __str__(self):
        return self.msg.format(len(self.exhausted))

def shortest_path(cmat, start=None, end=None, heuristic=None):
    '''
    finds the shortest path in a graph, given edge costs. can start at any
    point in [start] and end at any point in [end]
    '''
    if start is None:
        start = (0,)
    if end is None:
        end = (cmat.col.max(),)
    if heuristic is None:
        heuristic = lambda i: 0

    reached = set(start)
    exhausted = set()
    vertex_costs = [0 if i in start else float('inf') for i in range(len(cmat.col))]

    while not all(i in exhausted for i in end):
        # exit condition
        if not any(reached):
            if any(i in end for i in exhausted):
                break
            raise PathNotFound(exhausted)

        # source vertex index, where source is the lowest cost vertex available
        estimate = lambda i: vertex_costs[i] + heuristic(i)
        si = lowest_cost_index = min(reached, key=estimate)
        source_cost = vertex_costs[si]

        # examine adjacent paths by index
        for pi in (cmat.row==si).nonzero()[0]: # ensure target not exhausted?
            ni = neighbor_index = cmat.col[pi]
            if ni in exhausted:
                continue
            reached.add(ni)

            travel_cost = cmat.data[pi]
            vertex_costs[ni] = min(vertex_costs[ni], source_cost+travel_cost)

        reached.remove(si)
        exhausted.add(si)

    # reconstruct path
    best_exit = min(end, key=lambda i: vertex_costs[i])
    path = [best_exit]
    while not any(i in path for i in start):
        parents = cmat.row[cmat.col==path[-1]]
        best_parent = min(parents, key=lambda i: vertex_costs[i])
        path.append(best_parent)
    path.reverse()

    return path