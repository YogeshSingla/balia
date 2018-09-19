graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}

graph = {'A': set(['B', 'C', 'E']),
         'B': set(['D', 'A', 'E']),
         'C': set(['A', 'F', 'G']),
         'D': set(['B', 'E']),
         'E': set(['A', 'B', 'D']),
         'F': set(['C']),
         'G': set(['C'])}


# graph = { 'A' : ['B','C','E'],
#           'B' : ['D','A','E'],
#           'C' : ['A','F','G'],
#           'D' : ['B','E'],
#           'E' : ['A','B','D'],
#           'F' : ['C'],
#           'G' : ['C']}

def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            print(vertex)
            stack.extend(graph[vertex] - visited)
    return visited

dfs(graph, 'A') # {'E', 'D', 'F', 'A', 'C', 'B'}