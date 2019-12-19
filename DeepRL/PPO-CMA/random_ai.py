import tensorflow as tf
import random
import graphical, game
import numpy as np
from Agent import Agent


class MyTools:
    N_Rows = 10
    N_Cols = 8
    N_Dir = 2
    def __init__(self):
        self._calculate_state_actions_stats()

    @staticmethod
    def match_in_row_from_index(board_num, i, j, flag_print):
        count_equal_forward = 1
        count_equal_backward = 0
        c1 = board_num[i, j]
        if c1 < 0:
            return False

        if j + 1 < MyTools.N_Cols and c1 == board_num[i, j + 1]:
            count_equal_forward += 1
            if j + 2 < MyTools.N_Cols and c1 == board_num[i, j + 2]:
                count_equal_forward += 1

        if j - 1 >= 0 and c1 == board_num[i, j - 1]:
            count_equal_backward += 1
            if j - 2 >= 0 and c1 == board_num[i, j - 2]:
                count_equal_backward += 1
        if flag_print:
            print(count_equal_backward + count_equal_forward)

        if count_equal_backward + count_equal_forward >= 3:
            return True
        return False

    @staticmethod
    def match_in_col_from_index(board_num, i, j, flag_print):
        count_equal_forward = 1
        count_equal_backward = 0
        r1 = board_num[i, j]
        if r1 < 0:
            return False

        if i + 1 < MyTools.N_Rows and r1 == board_num[i + 1, j]:
            count_equal_forward += 1
            if i + 2 < MyTools.N_Rows and r1 == board_num[i + 2, j]:
                count_equal_forward += 1

        if i - 1 >= 0 and r1 == board_num[i - 1, j]:
            count_equal_backward += 1
            if i - 2 >= 0 and r1 == board_num[i - 2, j]:
                count_equal_backward += 1

        if flag_print:
            print(count_equal_backward + count_equal_forward)

        if count_equal_backward + count_equal_forward >= 3:
            return True
        return False

    @staticmethod
    def swap_col(board_num, row, col):
        tmp = board_num[row, col + 1]
        board_num[row, col + 1] = board_num[row, col]
        board_num[row, col] = tmp
        return board_num

    @staticmethod
    def swap_row(board_num, row, col):
        tmp = board_num[row + 1, col]
        board_num[row + 1, col] = board_num[row, col]
        board_num[row, col] = tmp
        return board_num

    @staticmethod
    def get_board_num(board):
        # print(board)
        board_num = np.zeros((MyTools.N_Rows, MyTools.N_Cols))
        col = 0
        row = 0
        for s in range(0, len(board)):
            if board[s] != '\n':
                board_num[row, col] = ord(board[s]) - ord('a')
                board_num[row, col] = -1 if board_num[row, col] < 0 else board_num[row, col]

                col += 1
            else:
                col = 0
                row += 1
        # print(board_num)
        return board_num

    @staticmethod
    def get_game_state(board, moves_left):
        board_num = MyTools.get_board_num(board)

        flag_print = False
        if flag_print:
            print(moves_left)
        col_feature = np.zeros((MyTools.N_Rows, MyTools.N_Cols))
        for row in range(0, MyTools.N_Rows):
            for col in range(0, MyTools.N_Cols - 1):
                # flag_print = False
                # if row == 3 and col == 3:
                #    flag_print = True

                board_num = MyTools.swap_col(board_num, row, col)
                if flag_print:
                    print(board_num)
                if (MyTools.match_in_row_from_index(board_num, row, col, flag_print) or
                        MyTools.match_in_col_from_index(board_num, row, col, flag_print) or
                        MyTools.match_in_row_from_index(board_num, row, col + 1, flag_print) or
                        MyTools.match_in_col_from_index(board_num, row, col + 1, flag_print)):
                    col_feature[row, col] = 1
                board_num = MyTools.swap_col(board_num, row, col)
                if flag_print:
                    print(board_num)
        # print(col_feature)

        row_feature = np.zeros((MyTools.N_Rows, MyTools.N_Cols))
        for row in range(0, MyTools.N_Rows - 1):
            for col in range(0, MyTools.N_Cols):
                # flag_print = False
                # if row == 3 and col == 3:
                #    flag_print = True

                board_num = MyTools.swap_row(board_num, row, col)
                if flag_print:
                    print(board_num)
                if (MyTools.match_in_row_from_index(board_num, row, col, flag_print) or
                        MyTools.match_in_col_from_index(board_num, row, col, flag_print) or
                        MyTools.match_in_row_from_index(board_num, row + 1, col, flag_print) or
                        MyTools.match_in_col_from_index(board_num, row + 1, col, flag_print)):
                    row_feature[row, col] = 1
                board_num = MyTools.swap_row(board_num, row, col)
                if flag_print:
                    print(board_num)
        # print(row_feature)

        ###### put features in the state
        state = np.zeros((MyTools.N_Rows * MyTools.N_Cols))

        c_state_index = 0
        for row in range(0, MyTools.N_Rows):
            for col in range(0, MyTools.N_Cols):
                v = 0
                if col_feature[row, col] and row_feature[row, col]:
                    v = 1
                elif col_feature[row, col]:
                    v = 2
                elif row_feature[row, col]:
                    v = 3
                state[c_state_index] = 2.0 * (v / 3.0) - 1.0
                c_state_index += 1
                #state[c_state_index] = 2.0 * ((board_num[row, col] + 1) / 5.0) - 1.0
                #c_state_index += 1

        # state[c_state_index] = moves_left / 25.0

        return state

    @staticmethod
    def get_action_from(move, use_discrete_action=False):
        if len(move) != 3:
            print("3 dim moves are needed")
            return -1

        use_discrete_action = use_discrete_action
        if use_discrete_action:
            action = np.array(move)

            if move[2]:
                action[2] = 1
            else:
                action[2] = 0

            out_action = (action[2]) * (MyTools.N_Rows * MyTools.N_Cols) + (action[0] * MyTools.N_Rows + action[1])
        else:
            action = np.array(move)

            action[0] = action[0] / float(MyTools.N_Cols - 1.0)
            action[1] = action[1] / float(MyTools.N_Rows - 1.0)
            if move[2]:
                action[2] = 1
            else:
                action[2] = 0

            out_action = 2.0 * action - 1.0
        return out_action

    @staticmethod
    def get_move_from(action, use_discrete_action=False):
        action = np.array(action)
        use_discrete_action = use_discrete_action
        if use_discrete_action:
            row_col = action % (MyTools.N_Rows * MyTools.N_Cols)

            _dir = int(action / (MyTools.N_Rows * MyTools.N_Cols))

            return ( int(row_col / MyTools.N_Rows), row_col % MyTools.N_Rows, _dir >= 1 )
        else:
            if action.shape.__len__() == 0 or len(action) != 3:
                print("need 3 dim actions")
                return ()
            _dir = False
            if action[2] > 0:
               _dir = True
            in_action = (action + 1.0) / 2.0
            col = int(in_action[0] * MyTools.N_Cols)
            row = int(in_action[1] * MyTools.N_Rows)

            return (col, row, _dir)

    @staticmethod
    def get_valid_actions():
        All_Actions = np.arange(0, MyTools.N_Rows * MyTools.N_Cols * MyTools.N_Dir)
        # print(All_Actions)

        remove_ids = []
        for i in range(0, MyTools.N_Rows):
            action_id = MyTools.get_action_from((MyTools.N_Cols - 1, i, False))
            remove_ids.append(action_id)

        for i in range(0, MyTools.N_Cols):
            action_id = MyTools.get_action_from((i, MyTools.N_Rows - 1, True))
            remove_ids.append(action_id)
        print("removed actions:", remove_ids)
        return np.delete(All_Actions, remove_ids, axis=0), remove_ids

    def _calculate_state_actions_stats(self):
        #All_Actions, remove_ids = MyTools.get_valid_actions()
        # print(All_Actions)

        board_sample_str = "da#bb#ac\n#bbccbbd\n#cd#a#d#\n#c#d#ddc\n##ba##bc\nacadc#c#\n#d##cc##\nc#cbdacd\ndca#d#b#\ndd#dccdb"
        state = MyTools.get_game_state(board_sample_str, 0)

        self.N_All_Actions = 3# len(All_Actions)
        self.N_State = len(state)

        ###################################################### test ############################################
        # test action conversion
        num_error_in_conversion = 0
        for i in range(0, 160):
            a = MyTools.get_move_from(i,True)
            ii = MyTools.get_action_from(a,True)
            if i != ii:
                num_error_in_conversion += 1
        print("number of errors happens in action conversion: ", num_error_in_conversion)

        print("removed for example:", MyTools.get_move_from(159, True), MyTools.get_move_from(73, True))

        print("Num of state: ", self.N_State, ", Num of actions: ", self.N_All_Actions)


class Trainer:
    def __init__(
            self,
            buff_size = 2048,
            max_steps = 1000000):
        self.m_tools = MyTools()

        # Simulation budget (steps) per iteration. This is the main parameter to tune.
        # 8k works for relatively simple environments like the OpenAI Gym Roboschool 2D Hopper.
        # For more complex problems such as 3D humanoid locomotion, try 32k or even 64k.
        # Larger values are slower but more robust.
        self.buff_size = buff_size
        self.batch_size = 256

        # Stop training after this many steps
        self.max_steps = max_steps

        # Init tensorflow
        self.sess = tf.InteractiveSession()

        # Create environment (replace this with your own simulator)
        print("Using Ubisoft simulation environment")

        # Create the agent using the default parameters for the neural network architecture
        self.agent = Agent(
            stateDim=self.m_tools.N_State,
            actionDim=self.m_tools.N_All_Actions,
            actionMin=np.array([-1.0, -1.0, -1.0]),
            actionMax=np.array([1.0, 1.0, 1.0])
        )

        # Finalize initialization
        tf.global_variables_initializer().run(session=self.sess)
        self.agent.init(self.sess)  # must be called after TensorFlow global variables init

        # Main training loop
        self.totalSimSteps = 0
        self.iterSimSteps = 0

    def collect_observations(self, board, move, score_delta, next_board, moves_left):
        state = MyTools.get_game_state(board, moves_left + 1)
        reward = score_delta / 100.0
        action = MyTools.get_action_from(move)
        n_state = MyTools.get_game_state(next_board, moves_left)
        done = (moves_left == 0) or (reward > 0)
        
        # Save the experience point
        self.agent.memorize(state, action, reward, n_state, done)

        # Bookkeeping
        self.iterSimSteps += 1

    def update_model(self):
        if self.iterSimSteps >= self.buff_size:
            # All episodes of this iteration done, update the agent and print results
            averageEpisodeReturn = self.agent.updateWithMemorized(self.sess, batchSize=self.batch_size, verbose=False)
            self.totalSimSteps += self.iterSimSteps
            print("Simulation steps {}, average episode return {}".format(self.totalSimSteps, averageEpisodeReturn))

            self.iterSimSteps = 0

    def predict_action(self, board, score, moves_left):
        state = MyTools.get_game_state(board, moves_left)

        predicted_move = self.agent.act(self.sess,state)

        return MyTools.get_move_from(predicted_move[0])

    def init_episode(self):
        pass

global trainer

def ai_callback(board, score, moves_left):
    global trainer

    predicted_move = trainer.predict_action(board, score, moves_left)
    #_dir = random.randint(0, 1) == 0
    #return (random.randint(0, 7 if _dir else 6), random.randint(0, 8 if _dir else 9), _dir)
    return predicted_move

def transition_callback(board, move, score_delta, next_board, moves_left):
    global trainer

    trainer.collect_observations(board, move, score_delta, next_board, moves_left)
    trainer.update_model()
    #pass # This can be used to monitor outcomes of moves

def end_of_game_callback(boards, scores, moves, final_score):
    global trainer

    trainer.init_episode()

    return True # True = play another, False = Done


if __name__ == '__main__':
    global trainer
    trainer = Trainer()

    speedup = 1000.0
    g = graphical.Game(ai_callback, transition_callback, end_of_game_callback, speedup)
    g.run()
