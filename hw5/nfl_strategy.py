import random

class NFLStrategy:
    def __init__(self, plays, prob):
        ''' Creates a game using the given list of possible outcomes
            and the given probability distribution of those outcomes.

            plays -- a list of lists of (yards-gained, ticks-elapsed, turnover)
                     tuples indexed by offensive action then defensive action
            prob -- a probability distribution over the tuples in plays[o][d]
        '''
        self._plays = plays
        self._prob = prob


    def initial_position(self):
        ''' Returns the initial position in this game as a
            (yards-to-score, downs-left, distance, ticks) tuple.
        '''
        return (80, 4, 10, 24)


    def offensive_playbook_size(self):
        ''' Returns the number of offensive actions available in this game. '''
        return len(self._plays)


    def defensive_playbook_size(self):
        ''' Returns the number of defensive actions available in this game. '''
        return len(self._plays[0])
    
    
    def result(self, pos, offensive_play):
        ''' Returns the position that results from the given offensive play
            selection from the given position as a
            (field-position, downs-left, distance, ticks) tuple, and the outcome
            of that play as a (yards-gained, ticks-elapsed, turnover) tuple.

            pos -- a tuple (field_pos, downs_left, distance, time_in_ticks)
            offensive_play -- the index of an offensive play
        '''
        play_outcome = self._outcome(offensive_play, random.randrange(len(self._plays[0])))
        return self._update(pos, play_outcome), play_outcome

    
    def _update(self, pos, play_outcome):
        ''' Returns the position that results from the given position
            and the result of a play.

            pos -- a tuple (field_pos, downs_left, distance, time_in_ticks)
            play_outcome a tuple  (yards-gained, ticks-elapsed, turnover)
        '''
        fieldPosition, downsLeft, distance, timeLeft = pos
        yardsGained, timeElapsed, turnover = play_outcome

        if turnover:
            return (fieldPosition, downsLeft, distance, 0)
        else:
            fieldPosition -= yardsGained
            distance -= yardsGained
            downsLeft -= 1
            timeLeft -= timeElapsed

            if timeLeft < 0:
                timeLeft = 0
        
            if fieldPosition > 99:
                # safety!
                return (99, 4, 10, 0)
            elif fieldPosition < 0:
                # touchdown!
                return (0, 4, 0, 0)

            if distance <= 0:
                # first down
                downsLeft = 4
                distance = min(10, fieldPosition)

            if downsLeft == 0:
                # turnover on downs
                return (fieldPosition, 4, distance, 0)

        return (fieldPosition, downsLeft, distance, timeLeft)

    
    def game_over(self, pos):
        ''' Determines if the given position represents a game-over position.
        
            pos -- a tuple (field_pos, down, distance, time)
        '''
        fieldPosition, downsLeft, distance, timeLeft = pos

        return fieldPosition == 0 or fieldPosition == 100 or downsLeft == 0 or timeLeft == 0


    def win(self, pos):
        ''' Determines if the given position represents a game-won position.
        
            pos -- a tuple (field_pos, down, distance, time)
        '''
        fieldPosition, downsLeft, distance, timeLeft = pos

        return fieldPosition == 0


    def _outcome(self, off_action, def_action):
        ''' Returns a randomly selected result for the given offensive
            and defensive actions.

            off_action -- the index of an offensive play
            def_action -- the index of an offensive play
        '''
        if off_action < 0 or off_action >= len(self._plays):
            raise ValueError("invalid offensive play index %d" % off_action)
        if def_action < 0 or def_action >= len(self._plays[off_action]):
            raise ValueError("invalid defensive play index %d" % def_action)

        r = random.random()
        cumulative = self._prob[0]
        i = 0
        while r > cumulative and i + 1 < len(self._prob):
            cumulative += self._prob[i + 1]
            i += 1
        return self._plays[off_action][def_action][i]

    
    def simulate(self, policy, n):
        ''' Simulates games using the given policy and returns the
            winning percentage for the policy.

            policy -- a function from game positions to offensive actions
        '''
        wins = 0
        play_count = 0
        for i in range(n):
            pos = self.initial_position()
            while not self.game_over(pos):
                play_count += 1
                pos, _ = self.result(pos, policy(pos))
            if self.win(pos):
                wins += 1
        return wins / n
