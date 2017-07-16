import cv2
import random
import numpy as np
import argparse
import tensorflow as tf

from game import wrapped_flappy_bird as game
from util.models import model, dueling_model
from util.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from util.schedules import LinearSchedule
from util.tf_util import scope_vars, update_target

def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Flappy bird")
    # Environment
    parser.add_argument("--game", type=str, default='bird', help="the name of the game being played")
    parser.add_argument("--n_actions", type=int, default=2, help="number of valid actions")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # training
    parser.add_argument("--gamma", type=float, default=0.9, help="decay rate of past observations")
    parser.add_argument("--frame_per_action", type=int, default=1, help="frame per action")
    parser.add_argument("--observe", type=int, default=1000, help="timesteps to observe before training")
    parser.add_argument("--explore", type=float, default=2000000., help="frames over which to anneal epsilon")
    parser.add_argument("--initial_eps", type=float, default=0.1, help="initial value of epsilon when training")
    parser.add_argument("--final_eps", type=float, default=0.0001, help="final value of epsilon")
    # Core DQN parameters
    parser.add_argument("--replay_buffer_size", type=int, default=50000, help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for Adam optimizer")
    parser.add_argument("--num_steps", type=int, default=int(2e8), help="total number of steps to run the environment for")
    parser.add_argument("--batch_size", type=int, default=32, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning_freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target_update_freq", type=int, default=10000, help="number of iterations between every target network update")
    # Bells and whistles
    parser.add_argument("--test", action="store_true", default=False, help="whether or not in the testing phase")
    parser.add_argument("--double", action="store_false", default=True, help="whether or not to use double DQN")
    parser.add_argument("--dueling", action="store_false", default=True, help="whether or not to use dueling network")
    parser.add_argument("--prioritized", action="store_false", default=True, help="whether or to use prioritized replay buffer")
    parser.add_argument("--prioritized_alpha", type=float, default=0.6, help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized_beta0", type=float, default=0.4, help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized_eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
    parser.add_argument("--prioritized_beta_iter", type=float, default=2000000., help="eps parameter for prioritized replay buffer")
    # Checkpoints
    parser.add_argument("--save_freq", type=int, default=10000, help="save model once every time this many iterations are completed")

    return parser.parse_args()

def preprocess(s_t, x_t1):
    x_t1 = cv2.cvtColor(cv2.resize(x_t1, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (80, 80, 1))
    # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
    s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
    return s_t1


def playGame():
    args = parse_args()
    args.initial_eps = 0.0001 if args.test else args.initial_eps
    if args.double:
        save_dir = "02DoubleDQN/" if not args.dueling else "02DoubleDuelingDQN/"
    else:
        save_dir = "01DQN/" if not args.dueling else "01DuelingDQN/"
    print("double:{}, dueling:{}, prioritized:{}\n".format(args.double, args.dueling, args.prioritized))

    sess = tf.InteractiveSession()
    # placeholders
    s = tf.placeholder("float", [None, 80, 80, 4], name="state")
    target = tf.placeholder("float", [None], name="target")
    action = tf.placeholder("float", [None, args.n_actions], name="action") # actions taken: [0, 1] or [1, 0]


    # -----dueling---------
    q_func = model(s, args.n_actions, scope="q_func") if not args.dueling else dueling_model(s, args.n_actions, scope="q_func")
    # -----dueling---------

    # -----double---------
    if args.double:
        q_func_vars = scope_vars("q_func")
        # target q network evaluation
        q_target= model(s, args.n_actions, scope="q_target") if not args.dueling else dueling_model(s, args.n_actions, scope="q_target")
        q_target_vars = scope_vars("q_target")
    # -----double---------

    # define the cost function
    readout_action = tf.reduce_sum(tf.multiply(q_func, action), axis=1)
    td_errors = target - readout_action
    cost = tf.reduce_mean(tf.square(td_errors))
    train_step = tf.train.AdamOptimizer(args.lr).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # -----prioritized replay---------
    # initialize replay memory
    if args.prioritized:
        replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, alpha=args.prioritized_alpha)
        beta_schedule = LinearSchedule(args.prioritized_beta_iter,
                                       initial_p=args.prioritized_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(args.replay_buffer_size)
    # -----prioritized replay---------

    ''' printing
    a_file = open("logs_" + args.game + "/readout.txt", 'w')
    h_file = open("logs_" + args.game + "/hidden.txt", 'w')
    '''

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(args.n_actions)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # s_t : 80 * 80 * 4

    # load networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks/" + save_dir)
    already_trained = 0
    if checkpoint and checkpoint.model_checkpoint_path:
        already_trained = checkpoint.model_checkpoint_path
        already_trained = int(already_trained[already_trained.find('dqn-') + 4:])
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    EpsilonSchedule = LinearSchedule(args.explore, args.final_eps, args.initial_eps)
    t = already_trained
    epsilon = EpsilonSchedule.value(t)
    while "flappy bird" != "angry bird":
        #-----double---------
        # whether update q_target
        if args.double and t % args.target_update_freq == 0:
            sess.run(update_target(q_func_vars, q_target_vars))
        # -----double---------

        # choose an action epsilon greedily
        Q_t = q_func.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([args.n_actions])
        action_index = 0
        if t % args.frame_per_action == 0:
            action_index = random.randrange(args.n_actions) if random.random() < epsilon else np.argmax(Q_t)
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        s_t1 = preprocess(s_t, x_t1_colored)

        # store the transition in D
        replay_buffer.add(s_t, a_t, r_t, s_t1, terminal)

        # only scale down epsilon if done observing
        if t > args.observe:
            epsilon = EpsilonSchedule.value(t - args.observe)

        # only train if done observing
        if t > args.observe + already_trained:
            # -----prioritized replay---------
            # sample a minibatch to train on
            if args.prioritized:
                experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(t-args.observe-already_trained))
                (s_j_batch, a_batch, r_batch, s_j1_batch, done_batch, weights, batch_idxes) = experience
            else:
                s_j_batch, a_batch, r_batch, s_j1_batch, done_batch = replay_buffer.sample(args.batch_size)
            # -----prioritized replay---------

            target_batch = []
            # -----double---------
            Q_j1_batch = q_target.eval(feed_dict={s: s_j1_batch}) if args.double else q_func.eval(feed_dict = {s : s_j1_batch})
            # -----double---------

            for i in range(0, args.batch_size):
                terminal = done_batch[i]
                # if terminal, only equals reward
                if terminal:
                    target_batch.append(r_batch[i])
                else:
                    target_batch.append(r_batch[i] + args.gamma * np.max(Q_j1_batch[i]))

            # -----prioritized replay---------
            if args.prioritized:
                td_errs = td_errors.eval(feed_dict={target: target_batch, action: a_batch, s: s_j_batch})
                new_priorities = np.abs(td_errs) + args.prioritized_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)
            # -----prioritized replay---------

            # perform gradient step
            train_step.run(feed_dict = {
                target : target_batch,
                action : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save
        if t % args.save_freq == 0:
            saver.save(sess, "saved_networks/" + save_dir + args.game + '-dqn', global_step = t)

        # display
        if t <= args.observe:
            state = "observe"
        elif t > args.observe and t <= args.observe + args.explore:
            state = "explore"
        else:
            state = "train"
        info_expr = 'TIMESTEP:{}, STATE:{}, EPSILON:{:6f}, ACTION{}, REWARD:{}, Q_MAX:{}'
        print(info_expr.format(t, state, epsilon, action_index, r_t, np.max(Q_t)))

        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

if __name__ == "__main__":
    playGame()
