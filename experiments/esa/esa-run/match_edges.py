"""Maximum-cardinality, minimum-delay event matching from the NASA scorer."""

def match_edges(edges, n, m):
    # Successive shortest augmenting paths. Integer weights encode delay first,
    # then favor inclusion of the earliest differing edge in lexicographic order.
    source, sink = n + m, n + m + 1
    graph = [[] for _ in range(sink + 1)]
    def add(u, v, capacity, cost):
        graph[u].append([v, len(graph[v]), capacity, cost])
        graph[v].append([u, len(graph[u]) - 1, 0, -cost])
        return graph[u][-1]
    for i in range(n):
        add(source, i, 1, 0)
    for j in range(m):
        add(n + j, sink, 1, 0)
    factor = 1 << len(edges)
    refs = []
    for k, (_, _, _, _, i, j, delay) in enumerate(edges):
        refs.append((i, j, delay, add(i, n + j, 1, delay * factor - (1 << (len(edges) - k - 1)))))
    while True:
        dist = [None] * len(graph)
        prev = [None] * len(graph)
        dist[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for u, outgoing in enumerate(graph):
                if dist[u] is None:
                    continue
                for ei, (v, _, cap, cost) in enumerate(outgoing):
                    candidate = dist[u] + cost
                    if cap and (dist[v] is None or candidate < dist[v]):
                        dist[v], prev[v], changed = candidate, (u, ei), True
            if not changed:
                break
        if dist[sink] is None:
            break
        v = sink
        while v != source:
            u, ei = prev[v]
            edge = graph[u][ei]
            edge[2] -= 1
            graph[v][edge[1]][2] += 1
            v = u
    return sorted((i, j, delay) for i, j, delay, edge in refs if not edge[2])
