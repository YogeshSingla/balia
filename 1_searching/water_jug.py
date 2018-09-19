"""
@author: kirito
"""
def possible_moves(state):
  moves = []
  x = state[0]
  y = state[1]
  #fill 3L jug
  moves.append([3,y])
  #fill 5L jug
  moves.append([x,5])
  #empty 3L jug
  moves.append([0,y])
  #empty 5L jug
  moves.append([x,0])
  #fill 5L from 3L jug
  if (x - (5 - y)) >= 0:
    moves.append([(x - (5 - y)) , 5 ])
  #fill 3L from 5L jug
  if (y - (3 - x)) >= 0:
    moves.append([3, (y - (3 - x))])
  #fill 5L from 3L jug and empty 3L jug
  if (x + y) <= 3:
    moves.append([(x + y) , 0 ])
  #fill 3L from 5L jug and empty 5L jug
  if (x + y) <= 5:
    moves.append([0 , (x + y) ])
  
  return moves

def dfs(start):
    stack = [start]
    visited = []
    
    while stack:
        #push neighbours of current node in stack
        current_node = stack.pop() #use pop() for stack-dfs
        if current_node not in visited:
            visited.append(current_node)
            for n in possible_moves(current_node):
              stack.append(n)
              #print(len(stack))
            print current_node,
            if current_node[1] == 4:
              break
            
print("\ndfs")
dfs([0,0])
