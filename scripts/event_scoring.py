"""Deterministic one-to-one event scoring on inclusive integer intervals."""


def score_events(truths, alarms, scored_start, scored_end):
    """Score (start, end) intervals within an inclusive common scored range.

    Input order is the source row order. Returned matched indices refer to the
    original input lists. Both types of interval are clipped to the range;
    truth exclusions and partial clips are recorded explicitly.
    """
    if scored_start > scored_end:
        raise ValueError('scored range must be nonempty')

    def clip(items):
        result, excluded, partial = [], [], []
        for i, (start, end) in enumerate(items):
            if start > end:
                raise ValueError('interval start exceeds end')
            a, b = max(start, scored_start), min(end, scored_end)
            if a > b:
                excluded.append(i)
            else:
                if (a, b) != (start, end):
                    partial.append(i)
                result.append((a, b, i))
        return sorted(result), excluded, partial

    ts, excluded, partial = clip(truths)
    als, excluded_alarms, partial_alarms = clip(alarms)
    edges = []
    for ti, (a, b, row) in enumerate(ts):
        for ai, (c, d, alarm_row) in enumerate(als):
            if a <= d and c <= b:
                edges.append((a, c, row, alarm_row, ti, ai, max(a, c) - a))
    edges.sort()

    # Successive shortest augmenting paths. Integer weights encode delay first,
    # then favor inclusion of the earliest differing edge in lexicographic order.
    n, m = len(ts), len(als)
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
    matches = sorted((ts[i][2], als[j][2], delay) for i, j, delay, edge in refs if not edge[2])
    tp = len(matches)
    fp, fn = len(als) - tp, len(ts) - tp
    return {'matches': matches, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'excluded_truths': excluded, 'partly_clipped_truths': partial,
            'excluded_alarms': excluded_alarms, 'partly_clipped_alarms': partial_alarms}
