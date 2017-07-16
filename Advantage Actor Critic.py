import cv2
import numpy as np
import argparse
import tensorflow as tf

from game import wrapped_flappy_bird as game
from util.models import Actor, Critic
from util.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from util.schedules import LinearSchedule


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Flappy bird")
    # Environment
    parser.add_argument("--game", type=str, default='bird', help="the name of the game being played")
    parser.add_argument("--n_actions", type=int, default=2, help="number of valid actions")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # training
    parser.add_argument("--gamma", type=float, default=0.9, help="decay rate of past observations")
    parser.add_argument("--epsilon", type=float, default=1e-5, help="decay rate of past observations")
    parser.add_argument("--observe", type=int, default=100, help="timesteps to observe before training")
    # Core DQN parameters
    parser.add_argument("--replay_buffer_size", type=int, default=50000, help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for Adam optimizer")
    parser.add_argument("--batch_size", type=int, default=2, help="number of transitions to optimize at the same time")
    # Bells and whistles
    parser.add_argument("--prioritized", action="store_true", default=False, help="whether or to use prioritized replay buffer")
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
    save_dir = "03AAC/"
    sess = tf.InteractiveSession()
    # actor
    # placeholders
    actor_critic_ph_s = tf.placeholder("float", [None, 80, 80, 4], name="state")
    actor_ph_action = tf.placeholder("float", [None, args.n_actions], name="action") # actions taken: [0, 1] or [1, 0]
    actor_ph_td_error = tf.placeholder("float", [None], name="target")

    # model
    actor_eval = Actor(actor_critic_ph_s, args.n_actions, scope="actor_eval")  # [None * n_actions]
    actor_eval_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')

    # loss
    #readout_action = tf.reduce_sum(tf.multiply(actor_eval, actor_ph_action), axis=1) # [None]
    log_prob = tf.reduce_sum(tf.log(actor_eval) * actor_ph_action, axis=1, keep_dims=True)
    actor_loss_basic = log_prob * actor_ph_td_error
    entropy = -tf.reduce_sum(actor_eval*tf.log(actor_eval), axis=1, keep_dims=True)
    actor_loss = 0.001 * entropy + actor_loss_basic
    actor_loss = tf.reduce_mean(-actor_loss)
    actor_train = tf.train.AdamOptimizer(args.lr).minimize(actor_loss)

    # critic
    # placeholders
    critic_ph_r = tf.placeholder(tf.float32, [None], 'r') # reward
    critic_ph_v_ = tf.placeholder(tf.float32, [None], "v_next") # next state value

    # model
    critic_eval = tf.squeeze(Critic(actor_critic_ph_s, scope="critic_eval"))

    # loss
    critic_td_err =  critic_ph_r + args.gamma * critic_ph_v_ - critic_eval
    critic_loss = tf.reduce_sum(tf.square(critic_td_err))
    critic_train = tf.train.AdamOptimizer(args.lr).minimize(critic_loss)

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
    t = already_trained
    while "flappy bird" != "angry bird":

        # choose an action with actor_eval
        a_t = np.zeros([args.n_actions])
        if t < args.observe:
            action_index = np.random.randint(2)
        else:
            act_prob = actor_eval.eval(feed_dict={actor_critic_ph_s: [s_t]})[0]
            action_index = np.random.choice(np.arange(len(act_prob)),p=act_prob)
        a_t[action_index] = 1

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        s_t1 = preprocess(s_t, x_t1_colored)

        # store the transition in D
        replay_buffer.add(s_t, a_t, r_t, s_t1, terminal)

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

            # critic update
            v_t1_batch = critic_eval.eval(feed_dict={actor_critic_ph_s: s_j1_batch})
            feed_dict = {actor_critic_ph_s: s_j_batch, critic_ph_r: r_batch, critic_ph_v_: v_t1_batch}
            td_error_batch, cri_loss, _ = sess.run([critic_td_err, critic_loss, critic_train], feed_dict)

            # actor update
            feed_dict = {actor_critic_ph_s: s_j_batch, actor_ph_action: a_batch, actor_ph_td_error: td_error_batch}
            _, act_loss, log_p = sess.run([actor_train, actor_loss, log_prob], feed_dict)

            # -----prioritized replay---------
            if args.prioritized:
                new_priorities = np.abs(td_error_batch) + args.prioritized_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)
            # -----prioritized replay---------

            # display
            info_expr = 'TIMESTEP:{}, ACTION:{}, ACTOR_LOSS:{}, CRITIC_LOSS:{}, log_prob:{}, REWARD:{}'
            print(info_expr.format(t, action_index, -act_loss, cri_loss, log_p, r_t))

        # update the old values
        s_t = s_t1
        t += 1

        # save
        if t % args.save_freq == 0:
            saver.save(sess, "saved_networks/" + save_dir + args.game + '-dqn', global_step = t)

        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

if __name__ == "__main__":
    playGame()
