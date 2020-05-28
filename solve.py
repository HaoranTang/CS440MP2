import numpy as np
from collections import defaultdict

# Algorithm X


def AlgorithmX(X, Y, solution=[]):

    if not X:
        s = list(solution)
        return s
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            s = AlgorithmX(X, Y, solution)
            if (s != []):
                return s
            deselect(X, Y, r, cols)
            solution.pop()

    return []


# Algorithm X select


def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

# Algorithm X deselect


def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)

# check if we can add


def can_add_pentominos(board, pent, coord):
    for row in range(len(pent)):
        for col in range(len(pent[0])):
            if coord[0]+row+1 > board.shape[0] or coord[1]+col+1 > board.shape[1] or board[coord[0]+row][coord[1]+col] != 0:
                return False
    return True
#add


def add_pentomino(board, pent, coord):
    board[coord[0]:coord[0] + pent.shape[0], coord[1]:coord[1] + pent.shape[1]] += pent
# map pents to coord on the board


def PentsToBoard(board, pents):
    pentdict = {}
    allpossiblepents = [[] for _ in range(len(pents))]
    for i, pent in enumerate(pents):
        rotate = pent
        explored = {}
        explored = defaultdict(lambda: 0, explored)
        for flip in range(2):
            flip = rotate
            for rot in range(4):
                if (explored[hash(str(flip))] == 0):
                    allpossiblepents[i].append(flip)
                    explored[hash(str(flip))] = 1

                flip = np.rot90(flip)
            rotate = np.fliplr(rotate)
    for pentnum, direction in enumerate(allpossiblepents):
        for pentdirection, pent in enumerate(direction):
            for coord, _ in np.ndenumerate(board):
                if can_add_pentominos(board, pent, coord):
                    board[coord[0]:coord[0] + pent.shape[0], coord[1]:coord[1] + pent.shape[1]] += pent
                    pentdict[(pentnum, pentdirection, coord)] = [(pentnum * -1) - 1] + [coord[0] * board.shape[1] + coord[1] for coord in np.argwhere(board > 0)]
                    board[coord[0]:coord[0] + pent.shape[0], coord[1]:coord[1] + pent.shape[1]] -= pent

    return pentdict


def solve(board, pents):
    newboard = np.zeros(board.shape)

    pentdict = PentsToBoard(newboard, pents)

    X = list(range(-len(pents), 0)) + [coord[0] * newboard.shape[1] + coord[1] for coord in np.argwhere(newboard == 0)]
    allpossiblepents = [[] for _ in range(len(pents))]
    for i, pent in enumerate(pents):
        rotate = pent
        explored = {}
        explored = defaultdict(lambda: 0, explored)
        for flip in range(2):
            flip = rotate
            for rot in range(4):
                if (explored[hash(str(flip))] == 0):
                    allpossiblepents[i].append(flip)
                    explored[hash(str(flip))] = 1

                flip = np.rot90(flip)
            rotate = np.fliplr(rotate)

    Xin = {j: set() for j in X}
    for i in pentdict:
        for j in pentdict[i]:
            Xin[j].add(i)
    sol = AlgorithmX(Xin, pentdict)

    solution = [(allpossiblepents[selectpent[0]][selectpent[1]], selectpent[2]) for selectpent in sol]

    # [add_pentomino(newboard, pent, coord) for (pent, coord) in solution]
    # print(newboard)

    return solution
