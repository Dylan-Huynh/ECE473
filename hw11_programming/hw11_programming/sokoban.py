import util
import os, sys
import datetime, time
import argparse
import signal, gc

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
            self.solved = all(problem.map[b[0]][b[1]].target for b in self.boxes())
        return self.solved
    def act(self, problem, act):
        if act in self.adj: return self.adj[act]
        else:
            val = problem.valid_move(self,act)
            self.adj[act] = val
            return val

    def box_is_cornered(self, map, box, targets, boxes):

        def row(distance):
            box_count = 1
            target_count = 0
            index = box[1] + 1
            while not map[box[0]][index].wall:
                if map[box[0] + distance][index].floor:
                    return None
                elif map[box[0]][index].target:
                    target_count += 1
                elif (box[0], index) in boxes:
                    box_count += 1
                index += 1
            index = box[1] - 1
            while not map[box[0]][index].wall:
                if map[box[0] + distance][index].floor:
                    return None
                elif map[box[0]][index].target:
                    target_count += 1
                elif (box[0], index) in boxes:
                    box_count += 1
                index -= 1

            if box_count > target_count:
                return True
            return None

        def column(distance):
            target_count = 0
            box_count = 1
            index = box[0] + 1
            while not map[index][box[1]].wall:
                if map[index][box[1] + distance].floor:
                    return None
                elif map[index][box[1]].target:
                    target_count += 1
                elif (index, box[1]) in boxes:
                    box_count += 1
                index += 1
            index = box[0] - 1
            while not map[index][box[1]].wall:
                if map[index][box[1] + distance].floor:
                    return None
                elif map[index][box[1]].target:
                    target_count += 1
                elif (index, box[1]) in boxes:
                    box_count += 1
                index -= 1

            if box_count > target_count:
                return True
            return None

        if box not in targets:
            if map[box[0] - 1][box[1]].wall and map[box[0]][box[1] - 1].wall:
                return True
            elif map[box[0] - 1][box[1]].wall and map[box[0]][box[1] + 1].wall:
                return True
            elif map[box[0] + 1][box[1]].wall and map[box[0]][box[1] - 1].wall:
                return True
            elif map[box[0] + 1][box[1]].wall and map[box[0]][box[1] + 1].wall:
                return True

            if map[box[0] - 1][box[1]].wall:
                if row(distance=-1):
                    return True
            elif map[box[0] + 1][box[1]].wall:
                if row(distance=1):
                    return True
            elif map[box[0]][box[1] - 1].wall:
                if column(distance=-1):
                    return True
            elif map[box[0]][box[1] + 1].wall:
                if column(distance=1):
                    return True

        return None

    def adj_box(self, box, all_boxes):
        adj = []
        for i in all_boxes:
            if box[0] - 1 == i[0] and box[1] == i[1]:
                adj.append({'box': i, 'direction': 'vertical'})
            elif box[0] + 1 == i[0] and box[1] == i[1]:
                adj.append({'box': i, 'direction': 'vertical'})
            elif box[1] - 1 == i[1] and box[0] == i[0]:
                adj.append({'box': i, 'direction': 'horizontal'})
            elif box[1] + 1 == i[1] and box[0] == i[0]:
                adj.append({'box': i, 'direction': 'horizontal'})
        return adj

    def box_is_trapped(self, map, box, targets, all_boxes):
        if self.box_is_cornered(map, box, targets, all_boxes):
            return True

        adj_boxes = self.adj_box(box, all_boxes)
        for i in adj_boxes:
            if box not in targets and i not in targets:
                if i['direction'] == 'vertical':
                    if map[box[0]][box[1] - 1].wall and map[i['box'][0]][i['box'][1] - 1].wall:
                        return True
                    elif map[box[0]][box[1] + 1].wall and map[i['box'][0]][i['box'][1] + 1].wall:
                        return True
                if i['direction'] == 'horizontal':
                    if map[box[0] - 1][box[1]].wall and map[i['box'][0] - 1][i['box'][1]].wall:
                        return True
                    elif map[box[0] + 1][box[1]].wall and map[i['box'][0] + 1][i['box'][1]].wall:
                        return True

        return None

    def deadp(self, problem):
        temp_boxes = self.data[1:]
        for box in list(temp_boxes):
            if self.box_is_trapped(problem.map, box, problem.targets, temp_boxes):
                self.dead = True
        return self.dead
    def all_adj(self, problem):
        if self.all_adj_cache is None:
            succ = []
            for move in 'udlr':
                valid, box_moved, nextS = self.act(problem, move)
                if valid:
                    succ.append((move, nextS, 1))
            self.all_adj_cache = succ
        return self.all_adj_cache

class MapTile:
    def __init__(self, wall=False, floor=False, target=False):
        self.wall = wall
        self.floor = floor
        self.target = target

def parse_move(move):
    if move == 'u': return (-1,0)
    elif move == 'd': return (1,0)
    elif move == 'l': return (0,-1)
    elif move == 'r': return (0,1)
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
    valid_chars = 'T#@+$*. '

    def __init__(self, map, dead_detection=False, a2=False):
        self.map = [[]]
        self.dead_detection = dead_detection
        self.init_player = (0,0)
        self.init_boxes = []
        self.numboxes = 0
        self.targets = []
        self.parse_map(map)

    # parse the input string into game map
    # Wall              #
    # Player            @
    # Player on target  +
    # Box               $
    # Box on target     *
    # Target            .
    # Floor             (space)
    def parse_map(self, input_str):
        coordinates = lambda: (len(self.map)-1, len(self.map[-1])-1)
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
                box = (row,col) in s.boxes()
                player = (row,col) == s.player()
                if box and target: print(DrawObj.BOX_ON, end='')
                elif player and target: print(DrawObj.PLAYER, end='')
                elif target: print(DrawObj.TARGET, end='')
                elif box: print(DrawObj.BOX_OFF, end='')
                elif player: print(DrawObj.PLAYER, end='')
                elif self.map[row][col].wall: print(DrawObj.WALL, end='')
                else: print(DrawObj.FLOOR, end='')
            print()

    # decide if a move is valid
    # return: (whether a move is valid, whether a box is moved, the next state)
    def valid_move(self, s, move, p=None):
        if p is None:
            p = s.player()
        dx,dy = parse_move(move)
        x1 = p[0] + dx
        y1 = p[1] + dy
        x2 = x1 + dx
        y2 = y1 + dy
        if self.map[x1][y1].wall:
            return False, False, None
        elif (x1,y1) in s.boxes():
            if self.map[x2][y2].floor and (x2,y2) not in s.boxes():
                return True, True, SokobanState((x1,y1),
                    [b if b != (x1,y1) else (x2,y2) for b in s.boxes()])
            else:
                return False, False, None
        else:
            return True, False, SokobanState((x1,y1), s.boxes())

    ##############################################################################
    # Problem 1: Dead end detection                                              #
    # Modify the function below. We are calling the deadp function for the state #
    # so the result can be cached in that state. Feel free to modify any part of #
    # the code or do something different from us.                                #
    # Our solution to this problem affects or adds approximately 50 lines of     #
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################
    # detect dead end
    def dead_end(self, s):
        if not self.dead_detection:
            return False
        return s.deadp(self)

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
    # code in the file in total. Your can vary substantially from this.          #
    ##############################################################################

    def floor_fill(self, problem, grid, current_path, path_list, x, y, visit_grid):
        boxes = problem.boxes()
        if grid[x][y].floor and not visit_grid[x][y]:  # will proc if the mapTile exists and is new
            visit_grid[x][y] = True

            if (x - 1, y) in boxes and not grid[x - 2][y].wall and (x - 2, y) not in boxes:
                path_list.append(current_path + 'u')
            if (x + 1, y) in boxes and not grid[x + 2][y].wall and (x + 2, y) not in boxes:
                path_list.append(current_path + 'd')
            if (x, y - 1) in boxes and not grid[x][y - 2].wall and (x, y - 2) not in boxes:
                path_list.append(current_path + 'l')
            if (x, y + 1) in boxes and not grid[x][y + 2].wall and (x, y + 2) not in boxes:
                path_list.append(current_path + 'r')

            if not grid[x - 1][y].wall and (x - 1, y) not in boxes and not visit_grid[x - 1][y]:
                self.floor_fill(problem, grid, current_path + 'u', path_list, x - 1, y, visit_grid)
            if not grid[x + 1][y].wall and (x + 1, y) not in boxes and not visit_grid[x + 1][y]:
                self.floor_fill(problem, grid, current_path + 'd', path_list, x + 1, y, visit_grid)
            if not grid[x][y - 1].wall and (x, y - 1) not in boxes and not visit_grid[x][y - 1]:
                self.floor_fill(problem, grid, current_path + 'l', path_list, x, y - 1, visit_grid)
            if not grid[x][y + 1].wall and (x, y + 1) not in boxes and not visit_grid[x][y + 1]:
                self.floor_fill(problem, grid, current_path + 'r', path_list, x, y + 1, visit_grid)
            return path_list

        return path_list

    def find_player(self, player, path):
        if len(path) == 0:
            return player
        else:
            move = path[0]
            if move == 'u':
                player = self.find_player(player, path[1:])
                player = (player[0] - 1, player[1])
            elif move == 'd':
                player = self.find_player(player, path[1:])
                player = (player[0] + 1, player[1])
            elif move == 'l':
                player = self.find_player(player, path[1:])
                player = (player[0], player[1] - 1)
            elif move == 'r':
                player = self.find_player(player, path[1:])
                player = (player[0], player[1] + 1)

        return player

    def expand(self, s):
        if self.dead_end(s):
            return []

        visit_grid = []
        for i in range(0, len(self.map)):
            for j in range(0, len(self.map[0])):
                visit_grid

        path_list = self.floor_fill(s, self.map, '', list(), s.data[0][0], s.data[0][1], visit_grid)

        new_states = []
        for path in path_list:
            new_player = self.find_player(s.data[0], path)

            box_index = list(s.data[1:]).index(new_player)
            new_boxes = list(s.data[1:])
            if path[-1] == 'u':
                new_boxes[box_index] = (new_boxes[box_index][0] - 1, new_boxes[box_index][1])
            elif path[-1] == 'd':
                new_boxes[box_index] = (new_boxes[box_index][0] + 1, new_boxes[box_index][1])
            elif path[-1] == 'l':
                new_boxes[box_index] = (new_boxes[box_index][0], new_boxes[box_index][1] - 1)
            elif path[-1] == 'r':
                new_boxes[box_index] = (new_boxes[box_index][0], new_boxes[box_index][1] + 1)

            new_states.append(path, SokobanState(new_player, new_boxes), len(path))

        return new_states

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
        def manhattan(x, y):
            man = abs(x[0] - y[0]) + abs(x[1] - y[1])
            return man

        dist = 0
        for b in s.boxes():
            if b not in self.problem.targets:
                for t in self.problem.targets:
                    if t not in s.boxes():
                        dist += manhattan(t, b)
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
        total = 0
        player = s.player()
        nbox = 231
        near = 0
        for b in s.boxes():
            nbox = min(nbox, abs(b[0] - player[0]) + abs(b[1] - player[1]))
            ntarget = 231
            for t in self.problem.targets:
                ntarget = min(ntarget, abs(b[0] - t[0]) + abs(b[1] - t[1]))
            assert ntarget != 2**31, "map is too big or no targets or something else went wrong"
            if ntarget == 0:
                continue
            move = (up, down, left, right) = (self.problem.map[b[0]][b[1] + 1].wall, self.problem.map[b[0]][b[1] - 1].wall,
                                              self.problem.map[b[0] - 1][b[1]].wall, self.problem.map[b[0] + 1][b[1]].wall)
            for i in move:
                if i: near += 1
            if near == 0:
                multiplier = 0.3
            elif near == 1:
                multiplier = .8
            elif near == 2:
                if (up and down) or (left and right):
                    multiplier = 20
            else:
                multiplier = 100
            total += ntarget * multiplier
        total += nbox
        return total

# solve sokoban map using specified algorithm
#  algorithm can be ucs a a2 fa fa2
def solve_sokoban(map, algorithm='ucs', dead_detection=False):
    # problem algorithm
    if 'f' in algorithm:
        problem = SokobanProblemFaster(map, dead_detection, '2' in algorithm)
    else:
        problem = SokobanProblem(map, dead_detection, '2' in algorithm)

    # search algorithm
    h = Heuristic(problem).heuristic2 if ('2' in algorithm) else Heuristic(problem).heuristic
    if 'a' in algorithm:
        search = util.AStarSearch(heuristic=h)
    else:
        search = util.UniformCostSearch()

    # solve problem
    search.solve(problem)
    if search.actions is not None:
        print('length {} soln is {}'.format(len(search.actions), search.actions))
    if 'f' in algorithm:
        return search.totalCost, search.actions, search.numStatesExplored
    else:
        return search.totalCost, search.actions, search.numStatesExplored

# let the user play the map
def play_map_interactively(map, dt=0.2):

    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'

    seq = ""
    i = 0
    visited=[state]

    os.system(clear)
    print()
    problem.print_state(state)

    while True:
        while i > len(seq)-1:
            try:
                seq += input('enter some actions (q to quit, digit d to undo d steps ): ')
            except EOFError:
                print()
                return

        os.system(clear)
        if seq!="":
            print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i+1:])
        problem.print_state(state)

        if seq[i] == 'q':
            return
        elif seq[i] in ['u','d','l','r']:
            time.sleep(dt)
            valid, _, new_state = problem.valid_move(state, seq[i])
            state = new_state if valid else state
            visited.append(state)
            os.system(clear)
            print(seq)
            problem.print_state(state)
            if not valid:
                print('Cannot move ' + seq[i] + ' in this state')
        elif seq[i].isdigit():
            i = max(-1, i - 1 - int(seq[i]))
            seq = seq[:i+1]
            visited = visited[:i+2]
            state = visited[i+1]
            os.system(clear)
            print(seq)
            problem.print_state(state)

        if state.is_goal(problem):
            for _ in range(10): print('\033[30;101mWIN!!!!!\033[0m')
            time.sleep(5)
            return
        i = i + 1

# animate the sequence of actions in sokoban map
def animate_sokoban_solution(map, seq, dt=0.2):
    problem = SokobanProblem(map)
    state = problem.start()
    clear = 'cls' if os.name == 'nt' else 'clear'
    for i in range(len(seq)):
        os.system(clear)
        print(seq[:i] + DrawObj.UNDERLINE + seq[i] + DrawObj.END + seq[i+1:])
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
                else: break
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

def extract_timeout(file, level):
    start = False
    found = False
    with open(file, 'r') as f:
        for line in f:
            if line[0] == "'": continue
            if line.strip().lower()[:5] == 'level':
                if start: break
                if line.strip().lower() == 'level ' + level:
                    found = True
                    continue
            if found:
                if line.strip().lower()[:7] == 'timeout':
                    return(int(line.strip().lower()[8:]))
                else: break
    if not found:
        raise Exception('Level ' + level + ' not found')
    return None

def solve_map(file, level, algorithm, dead, simulate):
    map = read_map_from_file(file, level)
    print(map)
    if dead: print('Dead end detection on for solution of level {level}'.format(**locals()))
    if algorithm == "me":
        play_map_interactively(map)
    else:
        tic = datetime.datetime.now()
        cost, sol, nstates = solve_sokoban(map, algorithm, dead)
        toc = datetime.datetime.now()
        print('Time consumed: {:.3f} seconds using {} and exploring {} states'.format(
            (toc - tic).seconds + (toc - tic).microseconds/1e6, algorithm, nstates))
        seq = ''.join(sol)
        print(len(seq), 'moves')
        print(' '.join(seq[i:i+5] for i in range(0, len(seq), 5)))
        if simulate:
            animate_sokoban_solution(map, seq)
        return (toc - tic).seconds + (toc - tic).microseconds/1e6

def main():
    parser = argparse.ArgumentParser(description="Solve Sokoban map")
    parser.add_argument("level", help="Level name or 'all'")
    parser.add_argument("algorithm", help="me | ucs | [f][a[2]] | all")
    parser.add_argument("-d", "--dead", help="Turn on dead state detection (default off)", action="store_true")
    parser.add_argument("-s", "--simulate", help="Simulate the solution (default off)", action="store_true")
    parser.add_argument("-f", "--file", help="File name storing the levels (levels.txt default)", default='levels.txt')
    parser.add_argument("-t", "--timeout", help="Seconds to allow (default 300) (ignored if level specifies)", type=int, default=300)

    args = parser.parse_args()
    level = args.level
    algorithm = args.algorithm
    dead = args.dead
    simulate = args.simulate
    file = args.file
    maxSeconds = args.timeout

    if (algorithm == 'all' and level == 'all'):
        raise Exception('Cannot do all levels with all algorithms')

    def solve_now(): return solve_map(file, level, algorithm, dead, simulate)

    def solve_with_timeout(timeout):
        level_timeout = extract_timeout(file,level)
        if level_timeout != None: timeout = level_timeout

        try:
            return util.TimeoutFunction(solve_now, timeout)()
        except KeyboardInterrupt:
            raise
        except MemoryError as e:
            signal.alarm(0)
            gc.collect()
            print('Memory limit exceeded.')
            return None
        except util.TimeoutFunctionException as e:
            signal.alarm(0)
            print('Time limit (%s seconds) exceeded.' % timeout)
            return None

    if level == 'all':
        levels = extract_levels(file)
        solved = 0
        time_used = 0
        for level in levels:
            print('Starting level {}'.format(level), file=sys.stderr)
            sys.stdout.flush()
            result = solve_with_timeout(maxSeconds)
            if result != None:
                solved += 1
                time_used += result
        print (f'\n\nOVERALL RESULT: {solved} levels solved out of {len(levels)} ({100.0*solved/len(levels)})% using {time_used:.3f} seconds')
    elif algorithm == 'all':
        for algorithm in ['ucs', 'a', 'a2', 'f', 'fa', 'fa2']:
            print('Starting algorithm {}'.format(algorithm), file=sys.stderr)
            sys.stdout.flush()
            solve_with_timeout(maxSeconds)
    elif algorithm == 'me':
        solve_now()
    else:
        solve_with_timeout(maxSeconds)

if __name__ == '__main__':
    main()
