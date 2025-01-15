import random
import time
import sys
import pygame
import math
import numpy as np

class connect4Player(object):
	def __init__(self, position, seed=0):
		self.position = position
		self.opponent = None
		self.seed = seed
		random.seed(seed)

	def play(self, env, move):
		move = [-1]

class human(connect4Player):

	def play(self, env, move):
		move[:] = [int(input('Select next move: '))]
		while True:
			if int(move[0]) >= 0 and int(move[0]) <= 6 and env.topPosition[int(move[0])] >= 0:
				break
			move[:] = [int(input('Index invalid. Select next move: '))]

class human2(connect4Player):

	def play(self, env, move):
		done = False
		while(not done):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

				if event.type == pygame.MOUSEMOTION:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					posx = event.pos[0]
					if self.position == 1:
						pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
					else: 
						pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
				pygame.display.update()

				if event.type == pygame.MOUSEBUTTONDOWN:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))
					move[:] = [col]
					done = True

class randomAI(connect4Player):

	def play(self, env, move):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		move[:] = [random.choice(indices)]

class stupidAI(connect4Player):

	def play(self, env, move):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		if 3 in indices:
			move[:] = [3]
		elif 2 in indices:
			move[:] = [2]
		elif 1 in indices:
			move[:] = [1]
		elif 5 in indices:
			move[:] = [5]
		elif 6 in indices:
			move[:] = [6]
		else:
			move[:] = [0]

class minimaxAI(connect4Player):
	def __init__(self, position, seed=0, depth_limit=2):
		self.position = position
		self.opponent = 3 - position
		self.seed = seed
		self.depth_limit = depth_limit
		self.board_score = np.array([[[1,25,46], [1,7,28,49], [1,7,13,31,52], [1,7,13,19,34,55,58], [7,13,19,37,61], [13,19,40,64], [19,43,67]],
									 [[2,25,26,47], [2,8,28,29,46,50], [2,8,14,31,32,49,53,58], [1,8,14,20,34,35,52,56,59,61], [8,14,20,37,38,55,62,64], [14,20,40,41,65,67], [20,43,44,68]],
									 [[3,25,26,27,48], [3,9,28,29,30,47,51,58], [3,9,15,31,32,33,46,50,54,59,61], [3,9,15,21,34,35,36,49,53,57,60,62,64], [9,15,21,37,38,39,52,56,63,65,67], [15,21,40,41,42,55,66,68], [21,43,44,45,69]],
									 [[4,25,26,27,58], [4,10,28,29,30,48,59,61], [4,10,16,31,32,33,47,51,60,62,64], [4,10,16,22,34,35,36,46,50,54,63,65,67], [10,16,22,37,38,39,49,53,57,66,68], [16,22,40,41,42,52,56,69], [22,43,44,45,55]],
									 [[5,26,27,59], [5,11,29,30,60,62], [5,11,17,32,33,48,63,65], [5,11,17,23,35,36,47,51,66,68], [11,17,23,38,39,50,54,69], [17,23,41,42,53,57], [23,44,45,56]],
									 [[6,27,60], [6,12,30,63], [6,12,18,33,66], [6,12,18,24,36,48,69], [12,18,24,39,51], [18,24,42,54], [24,45,57]]])
		self.maxima = np.iinfo(np.int16).max
		self.minima = np.iinfo(np.int16).min
		self.all_counts = 4*6 + 3*7 + 4*3 + 4*3
		random.seed(seed)

	def play(self, env, move):
		env.visualize = True
		depth = self.depth_limit
		player = self.position
		vs = np.zeros(7) + self.minima
		for a in self.actions(env):
			vs[a] = self.min_helper(env.getEnv(), a, player, depth-1)
		move[:] = [np.argmax(vs)]

	def max_helper(self, env, move, player, depth):
		env = self.simulateMove(env, move, player)
		if self.cutoff_test(env, move, player, depth):
			return self.eval(env, move, player, depth)
		v = self.minima
		for a in self.actions(env):
			v = np.max((v, self.min_helper(env.getEnv(), a, 3-player, depth-1)))
		if len(env.history[0]) == 42: return 0
		return v

	def min_helper(self, env, move, player, depth):
		env = self.simulateMove(env, move, player)
		if self.cutoff_test(env, move, player, depth):
			return self.eval(env, move, player, depth)
		v = self.maxima
		for a in self.actions(env):
			v = np.min((v, self.max_helper(env.getEnv(), a, 3-player, depth-1)))
		if len(env.history[0]) == 42: return 0
		return v

	def actions(self, env):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		return indices

	def cutoff_test(self, env, move, player, depth):
		if env.gameOver(move, player) or depth < 1:
			return True
		else:
			return False

	def eval(self, env, move, player, depth):  
		if env.gameOver(move, player):
			return self.utility(player)
		elif depth < 1:
			return self.heuristic(env)
		else:
			return 0

	def heuristic(self, env):
		board = env.getBoard()
		w, h = env.shape
		player = self.position
		opponent = 3 - player # self.opponent
		count_player = np.ones(self.all_counts+1)
		count_opponent = np.ones(self.all_counts+1)
		for i in range(w):
			for j in range(h):
				if board[i, j] == player:
					for item in self.board_score[i, j]:
						count_player[item] *= 2
						count_opponent[item] *= 0
				elif board[i, j] == opponent:
					for item in self.board_score[i, j]:
						count_player[item] *= 0
						count_opponent[item] *= 2
				else:
					continue
		score = np.sum(count_player) - np.sum(count_opponent)

		return score


	def simulateMove(self, env, move, player):
		env.board[env.topPosition[move]][move] = player
		env.topPosition[move] -= 1
		env.history[0].append(move)
		return env

	def utility(self, player):
		if player == self.position:
			return self.maxima
		elif player == 3-self.position:
			return self.minima

	def signal_handler(self):
		print("SIGTERM ENCOUNTERED")
		sys.exit(0)

class alphaBetaAI(connect4Player):
	def __init__(self, position, seed=0, depth_limit=2):
		self.position = position
		self.opponent = 3 - position
		self.seed = seed
		self.depth_limit = depth_limit
		self.board_score = np.array([[[1,25,46], [1,7,28,49], [1,7,13,31,52], [1,7,13,19,34,55,58], [7,13,19,37,61], [13,19,40,64], [19,43,67]],
									 [[2,25,26,47], [2,8,28,29,46,50], [2,8,14,31,32,49,53,58], [1,8,14,20,34,35,52,56,59,61], [8,14,20,37,38,55,62,64], [14,20,40,41,65,67], [20,43,44,68]],
									 [[3,25,26,27,48], [3,9,28,29,30,47,51,58], [3,9,15,31,32,33,46,50,54,59,61], [3,9,15,21,34,35,36,49,53,57,60,62,64], [9,15,21,37,38,39,52,56,63,65,67], [15,21,40,41,42,55,66,68], [21,43,44,45,69]],
									 [[4,25,26,27,58], [4,10,28,29,30,48,59,61], [4,10,16,31,32,33,47,51,60,62,64], [4,10,16,22,34,35,36,46,50,54,63,65,67], [10,16,22,37,38,39,49,53,57,66,68], [16,22,40,41,42,52,56,69], [22,43,44,45,55]],
									 [[5,26,27,59], [5,11,29,30,60,62], [5,11,17,32,33,48,63,65], [5,11,17,23,35,36,47,51,66,68], [11,17,23,38,39,50,54,69], [17,23,41,42,53,57], [23,44,45,56]],
									 [[6,27,60], [6,12,30,63], [6,12,18,33,66], [6,12,18,24,36,48,69], [12,18,24,39,51], [18,24,42,54], [24,45,57]]])
		self.maxima = np.iinfo(np.int16).max
		self.minima = np.iinfo(np.int16).min
		self.all_counts = 4*6 + 3*7 + 4*3 + 4*3
		random.seed(seed)

	def play(self, env, move):
		env.visualize = True
		depth = self.depth_limit
		player = self.position
		vs = np.zeros(7) + self.minima
		for a in self.actions(env):
			vs[a] = self.min_helper(env.getEnv(), a, player, depth-1, self.minima, self.maxima)
		move[:] = [np.argmax(vs)]

	def max_helper(self, env, move, player, depth, alpha, beta):
		env = self.simulateMove(env, move, player)
		if self.cutoff_test(env, move, player, depth):
			return self.eval(env, move, player, depth)
		v = self.minima
		for a in self.actions(env):
			v = np.max((v, self.min_helper(env.getEnv(), a, 3-player, depth-1, alpha, beta)))
			if v>=beta:
				return v
			alpha = max(a, v)
		if len(env.history[0]) == 42: return 0
		return v

	def min_helper(self, env, move, player, depth, alpha, beta):
		env = self.simulateMove(env, move, player)
		if self.cutoff_test(env, move, player, depth):
			return self.eval(env, move, player, depth)
		v = self.maxima
		for a in self.actions(env):
			v = np.min((v, self.max_helper(env.getEnv(), a, 3-player, depth-1, alpha, beta)))
			if v<=alpha:
				return v
			alpha = min(a, v)
		if len(env.history[0]) == 42: return 0
		return v

	def actions(self, env):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		return indices

	def cutoff_test(self, env, move, player, depth):
		if env.gameOver(move, player) or depth < 1:
			return True
		else:
			return False

	def eval(self, env, move, player, depth):  
		if env.gameOver(move, player):
			return self.utility(player)
		elif depth < 1:
			return self.heuristic(env)
		else:
			return 0

	def heuristic(self, env):
		board = env.getBoard()
		w, h = env.shape
		player = self.position
		opponent = 3 - player # self.opponent
		count_player = np.ones(self.all_counts+1)
		count_opponent = np.ones(self.all_counts+1)
		for i in range(w):
			for j in range(h):
				if board[i, j] == player:
					for item in self.board_score[i, j]:
						count_player[item] *= 2
						count_opponent[item] *= 0
				elif board[i, j] == opponent:
					for item in self.board_score[i, j]:
						count_player[item] *= 0
						count_opponent[item] *= 2
				else:
					continue
		score = np.sum(count_player) - np.sum(count_opponent)

		return score

	def simulateMove(self, env, move, player):
		env.board[env.topPosition[move]][move] = player
		env.topPosition[move] -= 1
		env.history[0].append(move)
		return env

	def utility(self, player):
		if player == self.position:
			return self.maxima
		elif player == 3-self.position:
			return self.minima

	def signal_handler(self):
		print("SIGTERM ENCOUNTERED")
		sys.exit(0)


SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)




