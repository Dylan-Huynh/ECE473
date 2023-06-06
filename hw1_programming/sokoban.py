import util
import os, sys
import datetime, time
import argparse


class SokobanState:
    # player: 2-tuple representing player location (coordinates)
    # boxes: list of 2-tuples indicating box locations
    def __init__(self, player, boxes):
        # self.data stores the state
        self.data = tuple([player] + sorted(boxes))
        # below are cache variables to avoid duplicated computation
        self.all_adj_cache = None
        self.adj = {}
        self.dead = None
        self.solved = None

    def __str__(self):
        return 'player: ' + str(self.player()) + ' boxes: ' + str(self.boxes())

    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data

    def __lt__(self, other):
        return self.data < other.data

    def __hash__(self):
        return hash(self.data)

    # return player location
    def player(self):
        return self.data[0]

    # return boxes locations
    def boxes(self):
        return self.data[1:]

    def is_goal(self, problem):
        if self.solved is None:
            # self.solved = all(b in problem.targets for b in self.boxes())
            self.solved = all(problem.map[b[0]][b[1]].target for b in self.boxes())
        return self.solved

    def act(self, problem, act):
        if act in self.adj:
            return self.adj[act]
        else:
            val = problem.valid_move(self, act)
            self.adj[act] = val
            return val

    def deadp(self, problem):
        if self.dead is None:
            raise NotImplementedError('Override me')
        problem.visited_data.append(self.data)
        problem.visited_dead.append(self.dead)
        return self.dead

    def all_adj(self, problem):
        if self.all_adj_cache is None:
            succ = []
            for move in 'udlr':
                valid, box_moved, states = self.act(problem, move)
                if valid:
                    succ.append((move, states, 1))
            self.all_adj_cache = succ
        return self.all_adj_cache


class MapTile:
    def __init__(self, wall=False, floor=False, target=False, dead=False):
        self.wall = wall
        self.floor = floor
        self.target = target
        self.dead = dead


def parse_move(move):
    if move == 'u':
        return (-1, 0)
    elif move == 'd':
        return (1, 0)
    elif move == 'l':
        return (0, -1)
    elif move == 'r':
        return (0, 1)
    raise Exception('Invalid move character.')


class DrawObj:
    WALL = '\033[37;47m \033[0m'
    PLAYER = '\033[97;40m@\033[0m'
    BOX_OFF = '\033[30;101mX\033[0m'
    BOX_ON = '\033[30;102mX\033[0m'
    TARGET = '\033[97;40m*\033[0m'
    FLOOR = '\033[30;40m \033[0m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class SokobanProblem(util.SearchProblem):
    # valid sokoban characters
    valid_chars = '#@+$*. '

    def __init__(self, map, dead_detection=False):
        self.map = [[]]
        self.dead_detection = dead_detection
        self.init_player = (0, 0)
        self.init_boxes = []
        self.numboxes = 0
        self.targets = []
        self.parse_map(map)
        self.parse_dead()
        # cache all visited state (map and dead)
        self.visited_data = []
        self.visited_dead = []

    # pre-determine which maptiles lead to deadstate
    def parse_dead(self):
        def deadstateExp(i, j, dir):
            if dir == 'r':
                b = j
                while self.map[i - 1][b].wall and not self.map[i][b].target:
                    # if basin structure found
                    if self.map[i][b + 1].wall:
                        for y in range(j + 1, b): self.map[i][y].dead = True
                        break
                    b += 1
            if dir == 'd':
                c = i
                while self.map[c][j - 1].wall and not self.map[c][j].target:
                    # if basin structure found
                    if self.map[c + 1][j].wall:
                        for x in range(i + 1, c): self.map[x][j].dead = True
                        break
                    c += 1

        # cycle through uldr
        dxy = ((-1, 0), (0, -1), (1, 0), (0, 1))
        for i in range(1, len(self.map) - 1):
            for j in range(1, len(self.map[i]) - 1):
                # compute if maptile is targetless floor
                if not self.map[i][j].dead and self.map[i][j].floor and (i, j) not in self.targets:
                    try:
                        # Freeze deadlock: at least one box is at a corner and not on a target
                        validity = [self.map[i + mv[0]][j + mv[1]].wall for mv in dxy]

                        # if up and left are walls
                        if validity[0] and validity[1]:
                            deadstateExp(i, j, 'r')
                            deadstateExp(i, j, 'd')

                        # if left and down are walls
                        if validity[1] and validity[2]:
                            deadstateExp(i, j, 'r')

                        # if up and right are walls
                        if validity[0] and validity[3]:
                            deadstateExp(i, j, 'd')

                    except IndexError as e:
                        self.map[i][j].dead = True
                    # if a box is cornered, it is considered unmovable and game is over
                    self.map[i][j].dead = any([validity[i] and validity[(i + 1) % 4] for i in range(4)])

    # parse the input string into game map
    # Wall              #
    # Player            @
    # Player on target  +
    # Box               $
    # Box on target     *
    # Target            .
    # Floor             (space)
    def parse_map(self, input_str):
        coordinates = lambda: (len(self.map) - 1, len(self.map[-1]) - 1)
        for c in input_str:
            if c == '#':
                self.map[-1].append(MapTile(wall=True))
            elif c == ' ':
                self.map[-1].append(MapTile(floor=True))
            elif c == '@':
                self.map[-1].append(MapTile(floor=True))
                self.init_player = coordinates()
            elif c == '+':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_player = coordinates()
                self.targets.append(coordinates())
            elif c == '$':
                self.map[-1].append(MapTile(floor=True))
                self.init_boxes.append(coordinates())
            elif c == '*':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.init_boxes.append(coordinates())
                self.targets.append(coordinates())
            elif c == '.':
                self.map[-1].append(MapTile(floor=True, target=True))
                self.targets.append(coordinates())
            elif c == '\n':
                self.map.append([])
        assert len(self.init_boxes) == len(self.targets), 'Number of boxes must match number of targets.'
        self.numboxes = len(self.init_boxes)

    def print_state(self, s):
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                target = self.map[row][col].target
                box = (row, col) in s.boxes()
                player = (row, col) == s.player()
                if box and target:
                    print(DrawObj.BOX_ON, end='')
                elif player and target:
                    print(DrawObj.PLAYER, end='')
                elif target:
                    print(DrawObj.TARGET, end='')
                elif box:
                    print(DrawObj.BOX_OFF, end='')
                elif player:
                    print(DrawObj.PLAYER, end='')
                elif self.map[row][col].wall:
                    print(DrawObj.WALL, end='')
                else:
                    print(DrawObj.FLOOR, end='')
            print()

    # decide if a move is valid
    # return: (whether a move is valid, whether a box is moved, the next state)
    def valid_move(self, s, move, p=None):
        if p is None:
            p = s.player()
        dx, dy = parse_move(move)
        x1 = p[0] + dx
        y1 = p[1] + dy
        x2 = x1 + dx
        y2 = y1 + dy
        if self.map[x1][y1].wall:
            return False, False, None
        elif (x1, y1) in s.boxes():
            if self.map[x2][y2].floor and (x2, y2) not in s.boxes():
                return True, True, SokobanState((x1, y1),
                                                [b if b != (x1, y1) else (x2, y2) for b in s.boxes()])
            else:
                return False, False, None
        else:
            return True, False, SokobanState((x1, y1), s.boxes())

    ##############################################################################
    # Problem 1: Dead end detection                                              #
    # Modify the function below. We are calling the deadp function for the state #
    # so the result can be cached in that state. Feel free to modify any part of #
    # the code or do something different from us.                                #
    # Our solution to this problem affects or adds approximately 50 lines of     #
    # code in the file in total. Yours can vary substantially from this.         #
    ##############################################################################
    # detect dead end
    def dead_end(self, s):
        if not self.dead_detection:
            return False

        return any(self.map[b[0]][b[1]].dead for b in s.boxes())

    def start(self):
        return SokobanState(self.init_player, self.init_boxes)

    def goalp(self, s):
        return s.is_goal(self)

    def expand(self, s):
        if self.dead_end(s):
            return []
        return s.all_adj(self)


class SokobanProblemFaster(SokobanProblem):
    ##############################################################################
    # Problem 2: Action compression                                              #
    # Redefine the expand function in the derived class so that it overrides the #
    # previous one. You may need to modify the solve_sokoban function as well to #
    # account for the change in the action sequence returned by the search       #
    # algorithm. Feel free to make any changes anywhere in the code.             #
    # Our solution to this problem affects or adds approximately 80 lines of     #
    # code in the file in total. Yours can vary substantially from this.         #
    ##############################################################################
    def expand(self, s):
        if self.dead_end(s):
            return []
        outcomes = []
        state = s

        # BFS
        visited = {}
        queue = []
        queue.append(s)
        visited[state] = True

        while queue:
            state = queue.pop(0)
            for d in 'udlr':
                val = state.act(self, d)
                if val[0] is False:
                    continue
                elif val[1] is True:
                    for b in state.boxes():
                        if val[2].player() == b:
                            player = list(b)
                            # location of the player before he pushes the box
                            if d == 'u':
                                player[0] += 1
                            elif d == 'd':
                                player[0] -= 1
                            elif d == 'l':
                                player[1] += 1
                            elif d == 'r':
                                player[1] -= 1
                            break
                    outcomes.append(((tuple(player), d), val[2], 1))
                else:
                    if val[2] not in visited:
                        queue.append(val[2])
                        visited[val[2]] = True
        return outcomes


class Heuristic:
    def __init__(self, problem):
        self.problem = problem

    ##############################################################################
    # Problem 3: Simple admissible heuristic                                     #
    # Implement a simple admissible heuristic function that can be computed      #
    # quickly based on Manhattan distance. Feel free to make any changes         #
    # anywhere in the code.                                                      #
    # Our solution to this problem affects or adds approximately 10 lines of     #
    # code in the file in total. Yours can vary substantially from this.         #
    ##############################################################################
    def heuristic(self, s):
        dist = 0

        # compute squared distance of each box not on target to each target without box
        for b in s.boxes():
            if b not in self.problem.targets:
                for t in self.problem.targets:
                    if t not in s.boxes():
                        dist += (t[0] - b[0]) ** 2 + (t[1] - b[1]) ** 2

        return dist

    ##############################################################################
    # Problem 4: Better heuristic.                                               #
    # Implement a better and possibly more complicated heuristic that need not   #
    # always be admissible, but improves the search on more complicated Sokoban  #
    # levels most of the time. Feel free to make any changes anywhere in the     #
    # code. Our heuristic does some significant work at problem initialization   #
    # and caches it.                                                             #
    # Our solution to this problem affects or adds approximately 40 lines of     #
    # code in the file in total. Yours can vary substantially from this.         #
    ##############################################################################
    def heuristic2(self, s):
        def getNearWallsMultiplier(map, x, y):
            boxes = s.boxes()
            tup = (up, down, left, right) = (
            map[x][y + 1].wall, map[x][y - 1].wall, map[x - 1][y].wall, map[x + 1][y].wall)
            surrounding = 0
            for square in tup:
                if square: surrounding += 1

            if surrounding == 0:
                return 0.3

            if surrounding == 1:
                return .8

            if surrounding == 2:
                # boxes look like this # #, they make a path, not an obstacle, but it still limits possibilities
                if (up and down) or (left and right):
                    return 20

            # really bad we're at a corner, shouldn't get here with dead end detection
            return 100

        total = 0
        player = s.player()
        closestBox = 2 ** 31

        for box in s.boxes():
            closestBox = min(closestBox, abs(box[0] - player[0]) + abs(box[1] - player[1]))
            # find closest target for that box
            closest = 2 ** 31
            for target in self.problem.targets:
                closest = min(closest, abs(box[0] - target[0]) + abs(box[1] - target[1]))

            assert closest != 2 ** 31, "map is too big or no targets or something else went wrong"
            # box already on target,
            if closest == 0:
                continue

            multiplier = getNearWallsMultiplier(self.problem.map, box[0], box[1])
            total += closest * multiplier

        total += closestBox
        return total


# solve sokoban map using specified algorithm
def solve_sokoban(map, algorithm='ucs', dead_detection=False):
    # problem algorithm
    if 'f' in algorithm:
        problem = SokobanProblemFaster(map, dead_detection)
    else:
        problem = SokobanProblem(map, dead_detection)

    # search algorithm
    h = Heuristic(problem).heuristic2 if ('2' in algorithm) else Heuristic(problem).heuristic
    if 'a' in algorithm:
        search = util.AStarSearch(heuristic=h)
    else:
        search = util.UniformCostSearch()

    # solve problem
    search.solve(problem)
    if search.actions is not None:
        if 'f' in algorithm:
            # decompression
            actions_decom = ''

            class tile:
                def __init__(self, action, currState, prevTile):
                    self.action = action
                    self.currState = currState
                    self.prevTile = prevTile

            state = problem.start()
            for action in search.actions:
                state_tile = tile(None, state, None)
                visited = {}
                queue = []
                queue.append(state_tile)
                visited[state] = True
                moves = ''

                while queue:
                    state_tile = queue.pop(0)
                    if state_tile.currState.player() == action[0]:
                        state = state_tile.currState
                        temp = state_tile
                        while (temp.prevTile != None):
                            moves = temp.action + moves
                            temp = temp.prevTile
                        moves = moves + action[1]
                        state = state.act(problem, action[1])[2]
                        break
                    for d in 'udlr':
                        val = state_tile.currState.act(problem, d)
                        if val[0] is False or val[1] is True:
                            continue
                        else:
                            if val[2] not in visited:
                                child = tile(d, val[2], state_tile)
                                queue.append(child)
                                visited[val[2]] = True
                actions_decom += moves
            search.actions = actions_decom

        print('length {} soln is {}'.format(len(search.actions), search.actions))
    if 'f' in algorithm:
        # change later
        return search.totalCost, search.actions, search.numStatesExplored
    else:
        return search.totalCost, search.actions, search.numStatesExplored


# animate the sequence of actions in sokoban map
def animate_sokoban_solution(map, seq, dt=0.2):
    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'
    for i in range(len(seq)):
        os.system(clear)
        print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i + 1:])
        problem.print_state(state)
        time.sleep(dt)
        valid, _, state = problem.valid_move(state, seq[i])
        if not valid:
            raise Exception('Cannot move ' + seq[i] + ' in state ' + str(state))
    os.system(clear)
    print(seq)
    problem.print_state(state)


# read level map from file, returns map represented as string
def read_map_from_file(file, level):
    map = ''
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip().lower()[:5] == 'level':
                if start: break
                if line.strip().lower() == 'level ' + level:
                    found = True
                    start = True
                    continue
            if start:
                if line[0] in SokobanProblem.valid_chars:
                    map += line
                else:
                    break
    if not found:
        raise Exception('Level ' + level + ' not found')
    return map.strip('\n')


# extract all levels from file
def extract_levels(file):
    levels = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip().lower()[:5] == 'level':
                levels += [line.strip().lower()[6:]]
    return levels


def solve_map(file, level, algorithm, dead, simulate):
    map = read_map_from_file(file, level)
    print(map)
    tic = datetime.datetime.now()
    cost, sol, nstates = solve_sokoban(map, algorithm, dead)
    toc = datetime.datetime.now()
    print('Time consumed: {:.3f} seconds using {} and exploring {} states'.format(
        (toc - tic).seconds + (toc - tic).microseconds / 1e6, algorithm, nstates))
    seq = ''.join(sol)
    print(len(seq), 'moves')
    print(' '.join(seq[i:i + 5] for i in range(0, len(seq), 5)))
    if simulate:
        animate_sokoban_solution(map, seq)


def main():
    parser = argparse.ArgumentParser(description="Solve Sokoban map")
    parser.add_argument("level", help="Level name or 'all'")
    parser.add_argument("algorithm", help="ucs | [f][a[2]] | all")
    parser.add_argument("-d", "--dead", help="Turn on dead state detection (default off)", action="store_true")
    parser.add_argument("-s", "--simulate", help="Simulate the solution (default off)", action="store_true")
    parser.add_argument("-f", "--file", help="File name storing the levels (levels.txt default)", default='levels.txt')
    parser.add_argument("-t", "--timeout", help="Seconds to allow (default 300)", type=int, default=300)

    args = parser.parse_args()
    level = args.level
    algorithm = args.algorithm
    dead = args.dead
    simulate = args.simulate
    file = args.file
    maxSeconds = args.timeout

    if (algorithm == 'all' and level == 'all'):
        raise Exception('Cannot do all levels with all algorithms')

    def solve_now():
        solve_map(file, level, algorithm, dead, simulate)

    def solve_with_timeout(maxSeconds):
        try:
            util.TimeoutFunction(solve_now, maxSeconds)()
        except KeyboardInterrupt:
            raise
        except MemoryError as e:
            signal.alarm(0)
            gc.collect()
            print('Memory limit exceeded.')
        except util.TimeoutFunctionException as e:
            signal.alarm(0)
            print('Time limit (%s seconds) exceeded.' % maxSeconds)

    if level == 'all':
        levels = extract_levels(file)
        for level in levels:
            print('Starting level {}'.format(level), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    elif algorithm == 'all':
        for algorithm in ['ucs', 'a', 'a2', 'f', 'fa', 'fa2']:
            print('Starting algorithm {}'.format(algorithm), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    else:
        solve_with_timeout(maxSeconds)


if __name__ == '__main__':
    main()