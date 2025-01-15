from copy import deepcopy
from queue import PriorityQueue
from Point import Point
import math

'''AIModule Interface
createPath(map map_) -> list<points>: Adds points to a path'''
class AIModule:

	def createPath(self, map_):
		pass

'''
A sample AI that takes a very suboptimal path.
This is a sample AI that moves as far horizontally as necessary to reach
the target, then as far vertically as necessary to reach the target.
It is intended primarily as a demonstration of the various pieces of the
program.
'''
class StupidAI(AIModule):

	def createPath(self, map_):
		path = []
		explored = []
		# Get starting point
		path.append(map_.start)
		current_point = deepcopy(map_.start)

		# Keep moving horizontally until we match the target
		while(current_point.x != map_.goal.x):
			# If we are left of goal, move right
			if current_point.x < map_.goal.x:
				current_point.x += 1
			# If we are right of goal, move left
			else:
				current_point.x -= 1
			path.append(deepcopy(current_point))

		# Keep moving vertically until we match the target
		while(current_point.y != map_.goal.y):
			# If we are left of goal, move right
			if current_point.y < map_.goal.y:
				current_point.y += 1
			# If we are right of goal, move left
			else:
				current_point.y -= 1
			path.append(deepcopy(current_point))

		# We're done!
		return path

class Djikstras(AIModule):

	def createPath(self, map_):
		q = PriorityQueue()
		cost = {}
		prev = {}
		explored = {}
		for i in range(map_.width):
			for j in range(map_.length):
				cost[str(i)+','+str(j)] = math.inf
				prev[str(i)+','+str(j)] = None
				explored[str(i)+','+str(j)] = False
		current_point = deepcopy(map_.start)
		current_point.comparator = 0
		cost[str(current_point.x)+','+str(current_point.y)] = 0
		q.put(current_point)
		while q.qsize() > 0:
			# Get new point from PQ
			v = q.get()
			if explored[str(v.x)+','+str(v.y)]:
				continue
			explored[str(v.x)+','+str(v.y)] = True
			# Check if popping off goal
			if v.x == map_.getEndPoint().x and v.y == map_.getEndPoint().y:
				break
			# Evaluate neighbors
			neighbors = map_.getNeighbors(v)
			for neighbor in neighbors:
				alt = map_.getCost(v, neighbor) + cost[str(v.x)+','+str(v.y)]
				if alt < cost[str(neighbor.x)+','+str(neighbor.y)]:
					cost[str(neighbor.x)+','+str(neighbor.y)] = alt
					neighbor.comparator = alt
					prev[str(neighbor.x)+','+str(neighbor.y)] = v
				q.put(neighbor)

		path = []
		while not(v.x == map_.getStartPoint().x and v.y == map_.getStartPoint().y):
			path.append(v)
			v = prev[str(v.x)+','+str(v.y)]
		path.append(map_.getStartPoint())
		path.reverse()
		return path

class AStarExp(AIModule):

	def heuristic(self, map_, p1, p2):
		dist = float(max(abs(p1.x-p2.x), abs(p1.y-p2.y)))
		if dist==0: 
			return 0
		h0 = float(map_.getTile(p1.x, p1.y))
		h1 = float(map_.getTile(p2.x, p2.y))
		return 2**((h1-h0)/dist)*dist

	def createPath(self, map_):
		q = PriorityQueue()
		cost = {}
		prev = {}
		explored = {}
		for i in range(map_.width):
			for j in range(map_.length):
				cost[str(i)+','+str(j)] = math.inf
				prev[str(i)+','+str(j)] = None
				explored[str(i)+','+str(j)] = False
		current_point = deepcopy(map_.start)
		current_point.comparator = 0
		cost[str(current_point.x)+','+str(current_point.y)] = 0
		q.put(current_point)
		while q.qsize() > 0:
			# Get new point from PQ
			v = q.get()
			if explored[str(v.x)+','+str(v.y)]:
				continue
			explored[str(v.x)+','+str(v.y)] = True
			# Check if popping off goal
			if v.x == map_.getEndPoint().x and v.y == map_.getEndPoint().y:
				break
			# Evaluate neighbors
			neighbors = map_.getNeighbors(v)
			for neighbor in neighbors:
				alt = map_.getCost(v, neighbor) + cost[str(v.x)+','+str(v.y)]
				if alt < cost[str(neighbor.x)+','+str(neighbor.y)]:
					cost[str(neighbor.x)+','+str(neighbor.y)] = alt
					neighbor.comparator = alt + self.heuristic(map_, neighbor, map_.getEndPoint())
					prev[str(neighbor.x)+','+str(neighbor.y)] = v
				q.put(neighbor)

		path = []
		while not(v.x == map_.getStartPoint().x and v.y == map_.getStartPoint().y):
			path.append(v)
			v = prev[str(v.x)+','+str(v.y)]
		path.append(map_.getStartPoint())
		path.reverse()
		return path

class AStarDiv(AIModule):

	def heuristic(self, map_, p1, p2):
		dist = float(max(abs(p1.x-p2.x), abs(p1.y-p2.y)))
		if dist==0: 
			return 0
		h0 = float(map_.getTile(p1.x, p1.y))
		h1 = float(map_.getTile(p2.x, p2.y))
		delta = (h1-h0)/dist
		return ((h0+h1)/2+delta)/((h0+h1)/2 + 1)*dist

	def createPath(self, map_):
		q = PriorityQueue()
		cost = {}
		prev = {}
		explored = {}
		for i in range(map_.width):
			for j in range(map_.length):
				cost[str(i)+','+str(j)] = math.inf
				prev[str(i)+','+str(j)] = None
				explored[str(i)+','+str(j)] = False
		current_point = deepcopy(map_.start)
		current_point.comparator = 0
		cost[str(current_point.x)+','+str(current_point.y)] = 0
		q.put(current_point)
		while q.qsize() > 0:
			# Get new point from PQ
			v = q.get()
			if explored[str(v.x)+','+str(v.y)]:
				continue
			explored[str(v.x)+','+str(v.y)] = True
			# Check if popping off goal
			if v.x == map_.getEndPoint().x and v.y == map_.getEndPoint().y:
				break
			# Evaluate neighbors
			neighbors = map_.getNeighbors(v)
			for neighbor in neighbors:
				alt = map_.getCost(v, neighbor) + cost[str(v.x)+','+str(v.y)]
				if alt < cost[str(neighbor.x)+','+str(neighbor.y)]:
					cost[str(neighbor.x)+','+str(neighbor.y)] = alt
					neighbor.comparator = alt + self.heuristic(map_, neighbor, map_.getEndPoint())
					prev[str(neighbor.x)+','+str(neighbor.y)] = v
				q.put(neighbor)

		path = []
		while not(v.x == map_.getStartPoint().x and v.y == map_.getStartPoint().y):
			path.append(v)
			v = prev[str(v.x)+','+str(v.y)]
		path.append(map_.getStartPoint())
		path.reverse()
		return path

class Astar():
	def __init__(self, map_):
		self.current_point = None
		self.target_point = None
		self.q = PriorityQueue()
		self.cost = {}
		self.prev = {}
		self.explored = {}
		for i in range(map_.width):
			for j in range(map_.length):
				self.cost[str(i)+','+str(j)] = math.inf
				self.prev[str(i)+','+str(j)] = None
				self.explored[str(i)+','+str(j)] = False

	def retrace_path(self, p):
		path = []
		v = self.current_point
		while not(v.x == p.x and v.y == p.y):
			path.append(v)
			v = self.prev[str(v.x)+','+str(v.y)]
		path.append(p)
		path.reverse()
		return path

class AStarMSH(AIModule):

	def heuristic(self, map_, p1, p2):
		dist = float(max(abs(p1.x-p2.x), abs(p1.y-p2.y)))
		if dist==0: 
			return 0
		h0 = float(map_.getTile(p1.x, p1.y))
		h1 = float(map_.getTile(p2.x, p2.y))
		return 2**((h1-h0)/dist)*dist

	def createPath(self, map_):
		fwp = Astar(map_)
		bwp = Astar(map_)

		start_point = deepcopy(map_.getStartPoint())
		end_point = deepcopy(map_.getEndPoint())
		start_point.comparator = 0
		end_point.comparator = 0

		fwp.cost[str(start_point.x)+','+str(start_point.y)] = 0
		bwp.cost[str(end_point.x)+','+str(end_point.y)] = 0
		fwp.q.put(start_point)
		bwp.q.put(end_point)
		while fwp.q.qsize() > 0 or bwp.q.qsize():
			fwp.current_point = fwp.q.get()
			bwp.current_point = bwp.q.get()
			# Check if popping off goal
			if fwp.current_point.x == bwp.current_point.x and fwp.current_point.y == bwp.current_point.y:
				break
			fwp.explored[str(fwp.current_point.x)+','+str(fwp.current_point.y)] = True
			bwp.explored[str(bwp.current_point.x)+','+str(bwp.current_point.y)] = True
			fwp.target_point = bwp.current_point
			bwp.target_point = fwp.current_point
			# Evaluate neighbors
			for astar in [fwp, bwp]:
				neighbors = map_.getNeighbors(astar.current_point)
				for neighbor in neighbors:
					if astar.explored[str(neighbor.x)+','+str(neighbor.y)]:
						continue
					alt = map_.getCost(astar.current_point, neighbor) + astar.cost[str(astar.current_point.x)+','+str(astar.current_point.y)]
					if alt < astar.cost[str(neighbor.x)+','+str(neighbor.y)]:
						astar.cost[str(neighbor.x)+','+str(neighbor.y)] = alt
						neighbor.comparator = alt + self.heuristic(map_, neighbor, astar.target_point)
						astar.prev[str(neighbor.x)+','+str(neighbor.y)] = astar.current_point
					astar.q.put(neighbor)
		fwpath = fwp.retrace_path(start_point)
		bwpath = bwp.retrace_path(end_point)
		bwpath.pop()
		bwpath.reverse()
		path = fwpath + bwpath
		return path

		

