from __future__ import division
import numpy as np
import nflgame
import random

# https://medium.com/hackernoon/ranking-nfl-teams-using-maximum-likelihood-estimation-7a4ed8994a67
# https://github.com/logicx24/NFLAnalysis

class NflMaximumLikelihoodEstimator:

	teams_array = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAC', 'JAX', 'KC', 'LA', 'LAC', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEN', 'WAS']

	"""
	current_weights: [0, ..., n]: n weights
	games_matrix = w[i][j]: games i played against j
	wins_array = [w_0, ..., w_n]: wins_array[i] = total wins by team i

	Function:

	R_a = (Number of Wins by A) / ( (Games A played against B) / (R_a + R_b) + ... + (Games A played against N) / (R_a + R_n))
	"""
	def optimization_function(self, current_weights, games_matrix, wins_array):
		out = np.array([i for i in current_weights])
		for i, curr_weight in enumerate(current_weights):
			running_sum = 0
			for j, opposing_weight in enumerate(current_weights):
				if curr_weight + opposing_weight > 0:
					running_sum += games_matrix[i][j] / (curr_weight + opposing_weight)
			if running_sum > 0:
				out[i] = wins_array[i] / running_sum
		return out


	"""
	Essentially, run numerical optimization on the above function until the weights stop changing.
	"""
	def iterate(self, games_matrix, wins_array, initial_weights=None):
		if not initial_weights:
			initial_weights = np.array([1.0] * len(wins_array))

		current_weights = np.array([0.0] * len(wins_array))
		next_weights = initial_weights
		while not np.allclose(current_weights, next_weights):
			current_weights, next_weights = next_weights, self.optimization_function(next_weights, games_matrix, wins_array)
		return next_weights


	def generate_matrices(self, season, weeks, kind="REG"):
		games = nflgame.games(season, week=weeks, kind=kind)
		games_matrix = np.identity(len(self.teams_array))

		team_index_map = {team : team_index for team_index, team in enumerate(self.teams_array)}
		game_wins_dict = {}

		for game in games:
			if game.winner not in game_wins_dict:
				game_wins_dict[game.winner] = 0
			game_wins_dict[game.winner] += self.wins_update_formula(game)

			games_matrix[team_index_map[game.winner]][team_index_map[game.loser]] += 1
			games_matrix[team_index_map[game.loser]][team_index_map[game.winner]] += 1

		wins_array = np.array([0] * len(self.teams_array))
		for team in game_wins_dict:
			wins_array[team_index_map[team]] = game_wins_dict[team]

		return games_matrix, wins_array

	def wins_update_formula(self, game):
		score_difference = abs(game.score_away - game.score_home)
		score_sum = game.score_away + game.score_home

		return 0.6 + (0.4 * (score_difference / score_sum))

	def generate_rankings(self, season, week):
		games_matrix, wins_array = self.generate_matrices(season, week)
		weights = self.iterate(games_matrix, wins_array)

		return {team : weights[i] for i, team in enumerate(self.teams_array)}


def main():
	w = NflMaximumLikelihoodEstimator()
	inter_weights = w.generate_rankings(2017, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

	print(inter_weights)
	print([x[0] for x in sorted(inter_weights.items(), key=lambda x: -x[1])])




if __name__ == "__main__":
	main()
	



def optimization_function(self, current_weights, games_matrix, wins_array):
	out = np.array([i for i in current_weights])
	for i, curr_weight in enumerate(current_weights):
		running_sum = 0
		for j, opposing_weight in enumerate(current_weights):
			if curr_weight + opposing_weight > 0:
				running_sum += games_matrix[i][j] / (curr_weight + opposing_weight)
		if running_sum > 0:
			out[i] = wins_array[i] / running_sum
	return out

def iterate(games_matrix, wins_array, initial_weights=None):
	if not initial_weights:
		initial_weights = np.array([1.0] * len(wins_array))

	current_weights = np.array([0.0] * len(wins_array))
	next_weights = initial_weights
	while not np.allclose(current_weights, next_weights):
		current_weights, next_weights = next_weights, optimization_function(next_weights, games_matrix, wins_array)
	return next_weights

import nflgame
teams_array = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAC', 'JAX', 'KC', 'LA', 'LAC', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEN', 'WAS']

def generate_matrices(season, weeks, kind="REG"):
	games = nflgame.games(season, week=weeks, kind=kind)
	games_matrix = np.identity(len(teams_array))

	team_index_map = {team : team_index for team_index, team in enumerate(teams_array)}
	game_wins_dict = {}

	for game in games:
		if game.winner not in game_wins_dict:
			game_wins_dict[game.winner] = 0
		game_wins_dict[game.winner] += 1

		games_matrix[team_index_map[game.winner]][team_index_map[game.loser]] += 1
		games_matrix[team_index_map[game.loser]][team_index_map[game.winner]] += 1

	wins_array = np.array([0] * len(teams_array))
	for team in game_wins_dict:
		wins_array[team_index_map[team]] = game_wins_dict[team]

	return games_matrix, wins_array

def generate_rankings(season, week):
    games_matrix, wins_array = generate_matrices(season, week)
    weights = iterate(games_matrix, wins_array)

    return {team : weights[i] for i, team in enumerate(teams_array)}

def wins_update_formula(game):
    score_difference = abs(game.score_away - game.score_home)
    score_sum = game.score_away + game.score_home

    return 0.5 + (0.5 * (score_difference / score_sum))

import nflgame
teams_array = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAC', 'JAX', 'KC', 'LA', 'LAC', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEN', 'WAS']

def generate_matrices(season, weeks, kind="REG"):
    games = nflgame.games(season, week=weeks, kind=kind)
    games_matrix = np.identity(len(teams_array))

    team_index_map = {team : team_index for team_index, team in enumerate(teams_array)}
    game_wins_dict = {}

    for game in games:
        if game.winner not in game_wins_dict:
            game_wins_dict[game.winner] = 0
        game_wins_dict[game.winner] += wins_update_formula(game)

        games_matrix[team_index_map[game.winner]][team_index_map[game.loser]] += 1
        games_matrix[team_index_map[game.loser]][team_index_map[game.winner]] += 1

    wins_array = np.array([0] * len(teams_array))
    for team in game_wins_dict:
        wins_array[team_index_map[team]] = game_wins_dict[team]

    return games_matrix, wins_array

[
 ('MIN', 7.873033209491705e-09),
 ('NE', 6.596693542005563e-09),
 ('PIT', 6.406584633672456e-09),
 ('PHI', 4.804992008728605e-09),
 ('NO', 3.97781703375456e-09),
 ('LA', 2.4346476927367584e-09),
 ('CAR', 2.264264287112924e-09),
 ('JAX', 1.3504237521036104e-09),
 ('ATL', 1.2669336267088476e-09),
 ('DAL', 1.2425271740905873e-09),
 ('KC', 1.2283568011803333e-09),
 ('LAC', 1.097401811766414e-09),
 ('BAL', 9.614213220211617e-10),
 ('DET', 9.278740058675194e-10),
 ('BUF', 7.43543700762921e-10),
 ('GB', 6.601783797154444e-10),
 ('SEA', 5.537923599090467e-10),
 ('WAS', 4.932473632791114e-10),
 ('CHI', 3.59274841968618e-10),
 ('TB', 3.5201922459685673e-10),
 ('TEN', 3.3955177585995183e-10),
 ('MIA', 3.082519794704152e-10),
 ('ARI', 2.9865368475689e-10),
 ('CIN', 2.81502023867731e-10),
 ('NYJ', 2.6356496117721624e-10),
 ('OAK', 2.357043123922683e-10),
 ('DEN', 1.8916856055648938e-10),
 ('SF', 1.719507938105613e-10),
 ('HOU', 5.112825542564584e-11),
 ('NYG', 4.657320748509618e-11),
 ('IND', 4.5743868177926426e-11),
 ('CLE', 0.0)
]