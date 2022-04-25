"""A quadrotor trajectory tracking example.

Notes:
    Includes and uses PID control.

Run as:

    $ python3 tracking.py --overrides ./tracking.yaml

"""
import time
import pybullet as p
from functools import partial

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

def main():
    """The main function creating, running, and closing an environment.

    """

    # Create an environment
    CONFIG_FACTORY = ConfigFactory()               
    config = CONFIG_FACTORY.merge()

    # Set iterations and episode counter.
    ITERATIONS = int(config.quadrotor_config['episode_len_sec']*config.quadrotor_config['ctrl_freq'])

    results_list = []

    controllers = ['mrac', 'lqr', 'lqr']
    # controllers = ['lqr', 'lqr', 'mrac']

    for i in range(3):
        # Start a timer.
        START = time.time()

        if i <= 1:
            pass
            config.quadrotor_config['disturbances']['dynamics'][0]['magnitude'] = [0, -0.3]
        else:
            config.quadrotor_config['disturbances']['dynamics'][0]['magnitude'] = [0, 0]

        # import pdb; pdb.set_trace()

        # Create controller.
        env_func = partial(make,
                           'quadrotor',
                           **config.quadrotor_config
                           )
        ctrl = make(controllers[i],
                    env_func,
                    plot_dir=f'results/plots/{i}/',
                    **config.quadrotor_config
                    )

        # import pdb; pdb.set_trace()

        results = ctrl.run()
        results_list.append(results)

        elapsed_sec = time.time() - START
        print("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n"
              .format(ITERATIONS, config.quadrotor_config.ctrl_freq, 1, elapsed_sec, ITERATIONS/elapsed_sec, (ITERATIONS*(1. / config.quadrotor_config.ctrl_freq))/elapsed_sec))

    for i, result in enumerate(results_list):
        print(f'Result for run {i}: {result}')


if __name__ == "__main__":
    main()
