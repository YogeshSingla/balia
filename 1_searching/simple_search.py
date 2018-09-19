# -*- coding: utf-8 -*-
"""
Created on Wed Aug 01 10:32:11 2018

@author: kirito

INCORRECT SOLUTION
Neighbours are being added and marked visited without visiting!
MESSED UP CODE!!!
"""

graph = { 'A' : ['B','C','E'],
          'B' : ['D','A','E'],
          'C' : ['A','F','G'],
          'D' : ['B','E'],
          'E' : ['A','B','D'],
          'F' : ['C'],
          'G' : ['C']}

def bfs(graph, start):
    queue = [start]
    visited = []
    
    while queue:
        #push neighbours of current node in queue
        current_node = queue.pop(0) #use pop() for stack-dfs
        visited.append(current_node)
        neighbours = graph[current_node]
        for n in neighbours:
            if(n not in visited):
                queue.append(n)
                visited.append(n)
        print current_node,

def dfs(graph, start):
    stack = [start]
    visited = []
    
    while stack:
        #push neighbours of current node in stack
        current_node = stack.pop() #use pop() for stack-dfs
        visited.append(current_node)
        neighbours = graph[current_node]
        for n in neighbours:
            if(n not in visited):
                stack.append(n)
                visited.append(n)
        print current_node,
        
print("bfs")
bfs(graph,'A')
print("\ndfs")
dfs(graph,'A')
