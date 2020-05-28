from time import sleep
from math import inf
import copy
import random

class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board = [['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_'],
                      ['_', '_', '_', '_', '_', '_', '_', '_', '_']]
        self.maxPlayer = 'X'
        self.minPlayer = 'O'
        self.maxDepth = 3
        # The start indexes of each local board
        self.globalIdx = [(0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6), (6, 0), (6, 3), (6, 6)]
        # Start local board index for reflex agent playing
        self.startBoardIdx = 4
        # self.startBoardIdx=randint(0,8)

        # utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility = 10000
        self.twoInARowMaxUtility = 500
        self.preventThreeInARowMaxUtility = 100
        self.cornerMaxUtility = 30

        self.winnerMinUtility = -10000
        self.twoInARowMinUtility = -100
        self.preventThreeInARowMinUtility = -500
        self.cornerMinUtility = -30

        self.winnerDesignUtility = 10000
        self.twoInARowDesignUtility = 280
        self.preventThreeInARowDesignUtility = 300
        self.cornerDesignUtility = 30
        self.maxEval = lambda: self.evaluatePredifined(True)
        self.minEval = lambda: self.evaluatePredifined(False)
        self.statesExplored = 0
        self.currPlayer=True
        self.DictofUtil = {}
        self.possiblerows = [(2, 0), (2, 2), (0, 0), (0, 2)]
        self.offsets = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]]))
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]]))
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]]))

    def setutility(self, arr, isMax):

        map = hash("".join(arr) + str(isMax))

        if map in (self.DictofUtil).keys():
            return (self.DictofUtil)[map]

        val = 0

        if not isMax:
            max2Unblocked = []
            for i in range(3):
                x = [self.minPlayer] * 2
                x.insert(i, '_')
                max2Unblocked.append(x)
            max1Unblocked = []
            for i in range(3):
                x = [self.maxPlayer] * 2
                x.insert(i, self.minPlayer)
                max1Unblocked.append(x)
            if arr == [self.minPlayer] * 3:
                val = self.winnerMinUtility
            elif arr in max2Unblocked:
                val = self.twoInARowMinUtility
            elif arr in max1Unblocked:
                val = self.preventThreeInARowMinUtility

        else:
            max2Unblocked = []
            for i in range(3):
                x = [self.maxPlayer] * 2
                x.insert(i, '_')
                max2Unblocked.append(x)
            max1Unblocked = []
            for i in range(3):
                x = [self.minPlayer] * 2
                x.insert(i, self.maxPlayer)
                max1Unblocked.append(x)
            if arr == [self.maxPlayer] * 3:
                val = self.winnerMaxUtility
            elif arr in max2Unblocked:
                val = self.twoInARowMaxUtility
            elif arr in max1Unblocked:
                val = self.preventThreeInARowMaxUtility

        (self.DictofUtil)[map] = val

        return val

    def copyLocalBoard(self, boardIdx):
        """
        This function returns a copy of local board of boardIdx for evaluation
        """
        r, c = self.globalIdx[boardIdx]
        local = [row[c:c + 3] for row in self.board[r:r + 3]]
        return local

    def newcheckrc(self, localBoard, totalUtil, UtilityofWinner, isMax):
        for i in range(3):
            u = self.setutility(localBoard[i], isMax)
            if u != UtilityofWinner:
                totalUtil += u
            else:
                return (UtilityofWinner, "win")
        for i in range(3):
            u = self.setutility([localBoard[r][i] for r in range(3)], isMax)
            if u != UtilityofWinner:
                totalUtil += u
            else:
                return (UtilityofWinner, "win")
        u = self.setutility([localBoard[i][2 - i] for i in range(3)], isMax)
        if u != UtilityofWinner:
            totalUtil += u
        else:
            return (UtilityofWinner, "win")
        u = self.setutility([localBoard[i][i] for i in range(3)], isMax)
        if u != UtilityofWinner:
            totalUtil += u
        else:
            return (UtilityofWinner, "win")
        return (totalUtil, "noresult")

    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """

        UtilityofWinner = self.winnerMaxUtility if isMax else self.winnerMinUtility
        totalUtil = 0

        boardlist1 = [self.copyLocalBoard(i) for i in range(len(self.globalIdx))]
        for localBoard in boardlist1:
            tuple = self.newcheckrc(localBoard, totalUtil, UtilityofWinner, isMax)
            if tuple[1] == "win":
                return UtilityofWinner
            elif tuple[1] == "noresult":
                totalUtil = tuple[0]

        if totalUtil == 0:
            if isMax:
                playermarks, UtilityatCorner = (self.maxPlayer, self.cornerMaxUtility)
            if not isMax:
                playermarks, UtilityatCorner = (self.minPlayer, self.cornerMinUtility)
            boardlist2 = [self.copyLocalBoard(i) for i in range(len(self.globalIdx))]
            for localBoard in boardlist2:
                for (row, column) in self.possiblerows:
                    if localBoard[row][column] == playermarks:
                        totalUtil += UtilityatCorner

        return totalUtil

    def ChooseValue(self, opponent, player, arr, max2Unblocked, max1Unblocked, isMax, map):
        connected3opponent = [opponent] * 3
        connected3player = [player] * 3
        if arr == connected3opponent:
            val = -self.winnerDesignUtility
        if arr == connected3player:
            val = self.winnerDesignUtility
        elif arr in max2Unblocked:
            val = self.twoInARowDesignUtility
        elif arr in max1Unblocked:
            val = self.preventThreeInARowDesignUtility

        if not isMax:
            val = -val

        (self.DictofUtil)[map] = val

        return val
    def ownUtility(self, arr, isMax):
        map = hash("".join(arr) + str(isMax))

        if map in (self.DictofUtil).keys():
            return (self.DictofUtil)[map]

        if not isMax:
            player = self.minPlayer
            opponent = self.maxPlayer
        else:
            player = self.maxPlayer
            opponent = self.minPlayer

        val = 0
        max2Unblocked = []
        for i in range(3):
            x = [player] * 2
            x.insert(i, '_')
            max2Unblocked.append(x)
        max1Unblocked = []
        for i in range(3):
            x = [opponent] * 2
            x.insert(i, player)
            max2Unblocked.append(x)
        return self.ChooseValue(opponent, player, arr, max2Unblocked, max1Unblocked, isMax, map)

    def checkrc(self, localBoard, isMax, totalUtil, UtilityofWinner, Utilityofloser):
        u = self.ownUtility([localBoard[i][i] for i in range(3)], isMax)
        if u != UtilityofWinner and u != Utilityofloser:
            totalUtil += u
        elif u == Utilityofloser:
            return ("lose", -1)
        elif u == UtilityofWinner:
            return ("win", 1)
        for i in range(3):
            u = self.ownUtility(localBoard[i], isMax)
            if u != UtilityofWinner and u != Utilityofloser:
                totalUtil += u
            elif u == Utilityofloser:
                return ("lose", -1)
            elif u == UtilityofWinner:
                return ("win", 1)

        u = self.ownUtility([localBoard[i][2 - i] for i in range(3)], isMax)
        if u != UtilityofWinner and u != Utilityofloser:
            totalUtil += u
        elif u == Utilityofloser:
            return ("lose", -1)
        elif u == UtilityofWinner:
            return ("win", 1)

        for i in range(3):
            u = self.ownUtility([localBoard[r][i] for r in range(3)], isMax)
            if u != UtilityofWinner and u != Utilityofloser:
                totalUtil += u
            elif u == Utilityofloser:
                return ("lose", -1)
            elif u == UtilityofWinner:
                return ("win", 1)
        return ("noresult", totalUtil)


    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """

        """
        Priority of our agent
        1) If we won, return the util for winning (10000 for isMax = True)
        2) If we lost, return the -util for losing (-10000 for isMax = True)
        3) Balanced scoring at 300 for each 2 in a row, and each prevent3inarow
        4)
        """
        UtilityofWinner = self.winnerDesignUtility if isMax else -self.winnerDesignUtility
        Utilityofloser = -UtilityofWinner
        totalUtil = 0

        for Boardcopy in (self.copyLocalBoard(i) for i in range(len(self.globalIdx))):
            tuple = self.checkrc(Boardcopy, isMax, totalUtil, UtilityofWinner, Utilityofloser)
            if tuple[0] == "win":
                return UtilityofWinner
            elif tuple[0] == "lose":
                return Utilityofloser
            totalUtil = tuple[1]

        if totalUtil == 0:
            if not isMax:
                playermarks, UtilityatCorner = (self.minPlayer, self.cornerMinUtility)
            else:
                playermarks, UtilityatCorner = (self.maxPlayer, self.cornerMaxUtility)
            for Boardcopy in (self.copyLocalBoard(i) for i in range(len(self.globalIdx))):
                for (row, column) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                    if Boardcopy[row][column] == playermarks:
                        totalUtil += UtilityatCorner

        return totalUtil

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        # YOUR CODE HERE
        for row in self.board:
            for v in row:
                if v == '_':
                    return True
        return False

    def checkWinner(self):
        # Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """

        for isMax in [True, False]:

            v = self.evaluatePredifined(isMax)
            if isMax and v == 10000:
                return 1
            elif not isMax:
                if v == -10000:
                    return -1

        return 0

    def Move(self, currBoardIdx, localRow, localCol, isMax, eraseMove=False):
        globalRow, globalCol = self.globalIdx[currBoardIdx]
        first = globalRow + localRow
        second = globalCol + localCol
        if eraseMove:
            self.board[first][second] = '_'
            return True
        if self.board[first][second] != '_':
            return False
        if isMax:
            self.board[first][second] = self.maxPlayer
            return True
        self.board[first][second] = self.minPlayer
        return True

    def alphabeta(self, depth, currBoardIdx, alpha, beta, isMax, returnCord=False):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """

        self.statesExplored += 1

        alreadywin = self.checkWinner()
        if depth == self.maxDepth or alreadywin != 0:
            if depth % 2 != 0:
                return self.minEval() if isMax else self.evaluatePredifined(True)
            else:
                return self.evaluatePredifined(True) if isMax else self.minEval()

        if isMax:
            value = -inf
        else:
            value = inf
        coordinate = (0, 0)

        for row in range(3):
            for column in range(3):
                if self.Move(currBoardIdx, row, column, isMax):
                    input1 = depth + 1
                    input2 = row * 3 + column
                    if not isMax:
                        value, coordinate = min([(value, coordinate),
                                                (self.alphabeta(input1, input2, alpha, beta, not isMax), (row, column))],
                                               key=lambda pair: pair[0])
                        if value >= alpha:
                            beta = value
                        else:
                            self.Move(currBoardIdx, row, column, isMax, eraseMove=True)
                            if returnCord:
                                return (value,coordinate)
                            return value
                    else:
                        value, coordinate = max([(value, coordinate),
                                                (self.alphabeta(input1, input2, alpha, beta, not isMax), (row, column))],
                                               key=lambda pair: pair[0])
                        if value <= beta:
                            alpha = value
                        else:
                            self.Move(currBoardIdx, row, column, isMax, eraseMove=True)
                            if returnCord:
                                return (value,coordinate)
                            return value
                    self.Move(currBoardIdx, row, column, isMax, eraseMove=True)

        return (value, coordinate) if returnCord else value

    def minimax(self, depth, currBoardIdx, isMax, returnCord=False):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        self.statesExplored += 1

        if depth == self.maxDepth or self.checkWinner() != 0:
            if depth % 2 != 0:
                if isMax:
                    return self.evaluatePredifined(False)
                return self.evaluatePredifined(True)

            else:
                if isMax:
                    return self.evaluatePredifined(True)
                return self.evaluatePredifined(False)


        if isMax:
            value = -inf
        else:
            value = inf
        coordinate = (0, 0)
        for row in range(3):
            for column in range(3):
                if self.Move(currBoardIdx, row, column, isMax):
                    input1 = depth + 1
                    input2 = row * 3 + column
                    if not isMax:
                        value, coordinate = min(
                            [(value, coordinate), (self.minimax(input1, input2, not isMax), (row, column))],
                            key=lambda pair: pair[0])
                    else:
                        value, coordinate = max(
                            [(value, coordinate), (self.minimax(input1, input2, not isMax), (row, column))],
                            key=lambda pair: pair[0])

                    self.Move(currBoardIdx, row, column, isMax, eraseMove=True)
        if returnCord:
            return (value, coordinate)
        return value


    def bigcheckWinner(self):
        original_board = copy.deepcopy(self.board)
        mini_board = []

        for i in range(9):
            local_winner = self.checksmallWinner(i, self.offsets)
            if local_winner == 1:
                mini_board.append('X')
            elif local_winner == -1:
                mini_board.append('O')
            else:
                mini_board.append('_')

        self.board = [['_'] * 9 for _ in range(9)]
        for i in range(9):
            offset = self.offsets[i]
            self.board[offset[0]][offset[1]] = mini_board[i]

        if self.count3(self.maxPlayer) > 0:
            result = 1
        elif self.count3(self.minPlayer) > 0:
            result = -1
        else:
            result = 0

        self.board = original_board
        return result

    def count3inrow(self, player_marker, local):
        count = 0
        if (local[0] == local[3] == local[6] and local[0] == player_marker):
            count += 1
        if (local[1] == local[4] == local[7] and local[1] == player_marker):
            count += 1
        if (local[2] == local[5] == local[8] and local[2] == player_marker):
            count += 1
        if (local[0] == local[1] == local[2] and local[0] == player_marker):
            count += 1
        if (local[3] == local[4] == local[5] and local[3] == player_marker):
            count += 1
        if (local[6] == local[7] == local[8] and local[6] == player_marker):
            count += 1
        return count
    def count3indiag(self, player_marker, local):
        count = 0
        if (local[0] == local[4] == local[8] and local[0] == player_marker):
            count += 1
        if (local[2] == local[4] == local[6] and local[2] == player_marker):
            count += 1
        return count
    def count3(self, player_marker):
        count = 0
        for idx in self.globalIdx:
            local = [self.board[idx[0] + 0][idx[1]],
                           self.board[idx[0] + 0][idx[1] + 1],
                           self.board[idx[0] + 0][idx[1] + 2],
                           self.board[idx[0] + 1][idx[1]],
                           self.board[idx[0] + 1][idx[1] + 1],
                           self.board[idx[0] + 1][idx[1] + 2],
                           self.board[idx[0] + 2][idx[1]],
                           self.board[idx[0] + 2][idx[1] + 1],
                           self.board[idx[0] + 2][idx[1] + 2]]

            count += self.count3inrow(player_marker, local)
            count += self.count3indiag(player_marker, local)
        return count

    def checksmallWinner(self, i, offsets):
        originalBoard = copy.deepcopy(self.board)
        idx = self.globalIdx[i]
        self.board = [['_'] * 9 for _ in range(9)]
        for offset in offsets:
            self.board[idx[0]+offset[0]][idx[1]+offset[1]] = originalBoard[idx[0]+offset[0]][idx[1]+offset[1]]
        winner = 0
        if self.count3(self.maxPlayer):
            winner = 1
        elif self.count3(self.minPlayer):
            winner = -1
        self.board = originalBoard
        return winner
    def twoinrow(self, local, player_marker):
        count = 0
        if (local[0] == local[1] == player_marker):
            if (local[2] == '_'):
                count += 1
        if (local[3] == local[4] == player_marker):
            if (local[5] == '_'):
                count += 1
        if (local[6] == local[7] == player_marker) and (local[8] == '_'):
            count += 1

        if (local[1] == local[2] == player_marker) and (local[0] == '_'):
            count += 1
        if (local[4] == local[5] == player_marker):
            if(local[3] == '_'):
                count += 1
        if (local[7] == local[8] == player_marker):
            if (local[6] == '_'):
                count += 1
        if (local[0] == local[2] == player_marker) and (local[1] == '_'):
            count += 1
        if (local[3] == local[5] == player_marker):
            if (local[4] == '_'):
                count += 1
        if (local[6] == local[8] == player_marker):
            if (local[7] == '_'):
             count += 1
        return count
    def twoincol(self, local, player_marker):
        count = 0
        if (local[0] == local[3] == player_marker):
            if (local[6] == '_'):
                count += 1
        if (local[1] == local[4] == player_marker):
            if (local[7] == '_'):
                count += 1
        if (local[2] == local[5] == player_marker):
            if (local[8] == '_'):
                count += 1

        if (local[3] == local[6] == player_marker):
            if (local[0] == '_'):
                count += 1
        if (local[4] == local[7] == player_marker):
            if (local[1] == '_'):
                count += 1
        if (local[5] == local[8] == player_marker):
            if (local[2] == '_'):
                count += 1

        if (local[0] == local[6] == player_marker) and (local[3] == '_'):
            count += 1
        if (local[5] == '_'):
            if (local[2] == local[8] == player_marker):
                count += 1
        if ((local[4] == '_') and (local[1] == local[7] == player_marker)):
            count += 1
        return count
    def twoindiag(self, local, player_marker):
        count = 0
        if (local[0] == local[8] == player_marker) and (local[4] == '_'):
            count += 1

        if (local[2] == local[6] == player_marker) and (local[4] == '_'):
            count += 1

        if (local[0] == local[4] == player_marker):
            if (local[8] == '_'):
                count += 1
        if (local[8] == local[4] == player_marker):
            if (local[0] == '_'):
                count += 1
        if (local[2] == local[4] == player_marker) and (local[6] == '_'):
            count += 1
        if (local[6] == local[4] == player_marker) and (local[2] == '_'):
            count += 1
        return count

    def count2(self, player_marker):
        count = 0
        for idx in self.globalIdx:
            local = [self.board[idx[0] + 0][idx[1]],
                           self.board[idx[0] + 0][idx[1] + 1],
                           self.board[idx[0] + 0][idx[1] + 2],
                           self.board[idx[0] + 1][idx[1]],
                           self.board[idx[0] + 1][idx[1] + 1],
                           self.board[idx[0] + 1][idx[1] + 2],
                           self.board[idx[0] + 2][idx[1]],
                           self.board[idx[0] + 2][idx[1] + 1],
                           self.board[idx[0] + 2][idx[1] + 2]]

            count += self.twoinrow(local, player_marker) + self.twoincol(local, player_marker) +self.twoindiag(local, player_marker)

        return count


    def playGameEC(self, maxFirst, startingBoard):
        bestMoves=[]
        bestValues=[]
        gameBoards=[]
        expandedNodesList=[]

        currIsMax = maxFirst
        currBoardIdx = startingBoard
        while (self.bigcheckWinner() == 0):
            if not self.checkMovesLeft():
                break

            self.expandedNodes = 0

            move_evaluations = []
            possiblemove = self.validmove(currBoardIdx)

            if len(possiblemove) == 0:
                break

            for move_coordinate in possiblemove:
                self.currPlayer = not currIsMax
                self.board[move_coordinate[0]][move_coordinate[1]] = self.maxPlayer if currIsMax else self.minPlayer
                evaluation = self.extracredit_alphabeta(3, move_coordinate[2], float('-inf'), float('inf'), currIsMax)
                move_evaluations.append((evaluation, move_coordinate[2], move_coordinate))
                self.board[move_coordinate[0]][move_coordinate[1]] = '_'

            if currIsMax:
                best_move = max(move_evaluations, key=lambda eval: (eval[0], -eval[1]))
            else:
                best_move = min(move_evaluations)
            self.changevalue(bestMoves, gameBoards, expandedNodesList, bestValues, best_move, currBoardIdx)

            self.board[best_move[2][0]][best_move[2][1]] = (self.maxPlayer if currIsMax else self.minPlayer)
            currIsMax = not currIsMax
            currBoardIdx = best_move[1]


        winner = self.bigcheckWinner()
        gameBoards.append(copy.deepcopy(self.board))

        return gameBoards, bestMoves, expandedNodesList, bestValues, winner

    def changevalue(self, bestMoves, gameBoards, expandedNodesList, bestValues, bestmove, currBoardIdx):
        bestMoves.append((currBoardIdx, bestmove[2]))
        gameBoards.append(copy.deepcopy(self.board))
        expandedNodesList.append(self.expandedNodes)
        bestValues.append(bestmove[0])

    def extracredit_alphabeta(self,depth,currBoardIdx,alpha,beta,isMax):

        self.expandedNodes += 1

        if depth == 1 or not self.checkMovesLeft() or self.bigcheckWinner() != 0:
            check = self.bigcheckWinner()
            if check != 0:
                return float('inf') * check
            value = 0
            value += self.count3(self.maxPlayer) * 5000
            value += self.count2(self.maxPlayer) * 100
            value -= self.count3(self.minPlayer) * 2500
            value -= self.count2(self.minPlayer) * 50
            return value

        if self.currPlayer:
            value = float('-inf')
            for move in self.validmove(currBoardIdx):
                self.board[move[0]][move[1]] = self.maxPlayer
                self.currPlayer = False
                new_value = self.extracredit_alphabeta(depth-1, move[2], alpha, beta, isMax)
                value = max(value, new_value)
                alpha = max(alpha, value)
                self.board[move[0]][move[1]] = '_'
                if (alpha >= beta):
                    break
            return value
        else:
            value = float('inf')
            for move in self.validmove(currBoardIdx):
                self.board[move[0]][move[1]] = self.minPlayer
                self.currPlayer = True
                new_value = self.extracredit_alphabeta(depth-1, move[2], alpha, beta, isMax)
                value = min(value, new_value)
                beta = min(beta, value)
                self.board[move[0]][move[1]] = '_'
                if (alpha >= beta):
                    break
            return value

    def validmove(self, currBoardIdx):
        possiblemove = []
        if self.checksmallWinner(currBoardIdx, self.offsets) == 0:
            for i in range(9):
                offset = self.offsets[i]
                index = self.globalIdx[currBoardIdx]
                move_coordinate = (index[0] + offset[0], index[1] + offset[1])
                marker = self.board[index[0] + offset[0]][index[1] + offset[1]]
                if marker == '_':
                    possiblemove.append((move_coordinate[0], move_coordinate[1], i))
        if len(possiblemove) == 0:
            for boardno in range(9):
                if self.checksmallWinner(boardno, self.offsets) == 0:
                    for i in range(9):
                        offset = self.offsets[i]
                        index = self.globalIdx[boardno]
                        marker = self.board[index[0] + offset[0]][index[1] + offset[1]]
                        if marker == '_':
                            possiblemove.append((index[0] + offset[0], index[1] + offset[1], i))

        return possiblemove


    def ChooseAgent(self, isMinimaxDefensive, isMinimaxOffensive, boardIdx):
        if isMinimaxDefensive:
            minagent = lambda boardIdx: self.minimax(0, boardIdx, False, returnCord=True)
            if isMinimaxOffensive:
                maxagent = lambda boardIdx: self.minimax(0, boardIdx, True, returnCord=True)
            else:
                maxagent = lambda boardIdx: self.alphabeta(0, boardIdx, -inf, inf, True, returnCord=True)
        else:
            minagent = lambda boardIdx: self.alphabeta(0, boardIdx, -inf, inf, False, returnCord=True)
            if isMinimaxOffensive:
                maxagent = lambda boardIdx: self.minimax(0, boardIdx, True, returnCord=True)
            else:
                maxagent = lambda boardIdx: self.alphabeta(0, boardIdx, -inf, inf, True, returnCord=True)
        return maxagent, minagent

    def playGamePredifinedAgent(self, maxFirst, isMinimaxOffensive, isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinateinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        bestMove = []
        bestValue = []
        expandedNodes = []
        gameBoards = []
        boardIdx = self.startBoardIdx
        maxagent, minagent = self.ChooseAgent(isMinimaxDefensive, isMinimaxOffensive,self.startBoardIdx)
        self.minEval = lambda: self.evaluatePredifined(False)
        self.maxEval = lambda: self.evaluatePredifined(True)
        while self.checkWinner() == 0:
            if not self.checkMovesLeft():
                break
            if maxFirst:
                v, (row, col) = maxagent(boardIdx)
            else:
                v, (row, col) = minagent(boardIdx)
            self.Move(boardIdx, row, col, maxFirst)
            bestMove.append((row, col))
            bestValue.append(v)
            expandedNodes.append(self.statesExplored)
            self.statesExplored = 0
            gameBoards.append(list(self.board))

            boardIdx = row * 3 + col
            maxFirst = not maxFirst

        return gameBoards, bestMove, expandedNodes, bestValue, self.checkWinner()

    def playGameYourAgent(self):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinateinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        maxFirst = random.choice([True, False])
        self.startBoardIdx = random.randint(0, 8)
        bestMove = []
        bestValue = []
        expandedNodes = []
        gameBoards = []
        boardIdx = self.startBoardIdx
        maxagent = lambda boardIdx: self.alphabeta(0, boardIdx, -10000, 10000, True, returnCord=True)
        minagent = lambda boardIdx: self.alphabeta(0, boardIdx, -10000, 10000, False, returnCord=True)
        self.maxEval = lambda: self.evaluatePredifined(True)
        self.minEval = lambda: self.evaluateDesigned(False)
        while self.checkMovesLeft():
            if self.checkWinner() != 0:
                break
            if maxFirst:
                val, (row, column) = maxagent(boardIdx)
            else:
                val, (row, column) = minagent(boardIdx)
            self.Move(boardIdx, row, column, maxFirst)
            bestMove.append((row, column))
            bestValue.append(val)
            expandedNodes.append(self.statesExplored)
            self.statesExplored = 0
            gameBoards.append(list(self.board))

            boardIdx = row * 3 + column
            maxFirst = not maxFirst

        return gameBoards, bestMove, self.checkWinner()

    def player(self):
        self.printGameBoard()
        i = None
        while i == None or len(i) != 2:
            i = [int(j) for j in input("choose the location(a,b)").split(",")]
        return 0, (i[1], i[0])
    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinateinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        # YOUR CODE HERE

        maxagent = lambda boardIdx: self.player()
        minagent = lambda boardIdx: self.alphabeta(0, boardIdx, -10000, 10000, False, returnCord=True)

        self.maxEval = lambda: self.evaluatePredifined(True)
        self.minEval = lambda: self.evaluateDesigned(False)
        boardIdx = self.startBoardIdx
        bestMove = []
        bestValue = []
        gameBoards = []
        expandedNodes = []
        maxFirst = random.choice([True, False])
        while self.checkMovesLeft() and self.checkWinner() == 0:
            print("Current Board: " + str(boardIdx))
            if maxFirst:
                v, (row, col) = maxagent(boardIdx)
            else:
                v, (row, col) = minagent(boardIdx)
            self.Move(boardIdx, row, col, maxFirst)
            bestMove.append((row, col))
            bestValue.append(v)
            expandedNodes.append(self.statesExplored)
            self.statesExplored = 0
            gameBoards.append(list(self.board))
            boardIdx = row * 3 + col
            maxFirst = not maxFirst

        return gameBoards, bestMove, self.checkWinner()

if __name__ == "__main__":
    uttt=ultimateTicTacToe()
    gameBoards, bestMove, winner = uttt.playGameHuman()
    print("Winner: " + str(winner) + " in " + str(len(bestMove)) + " turns. ")

    uttt.printGameBoard()
