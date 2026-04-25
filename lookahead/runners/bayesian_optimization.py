import numpy as np
from lookahead.model.gaussian_process import GaussianProcessSimple as GaussianProcess
import os
import logging

class BayesianOptimization(object):
    def __init__(self, search_space):
        self.opt_name = 'abstract'
        self.gaussian_process = None
        self.search_space = search_space

    def run(self, f, seed, budget_minus_initialization, initial_xs,
            checkpoint_dir=None, checkpoint_interval=50):

        # Load data for consistency with our other experiments
        np.random.seed(seed)
        d = len(self.search_space.domain_bounds)
        xhist = initial_xs
        yhist = f(xhist)
        self.gaussian_process = GaussianProcess(xhist, yhist)
        self.gaussian_process.train()

        while budget_minus_initialization > 0:
            logging.info(f"Budget remaining: {budget_minus_initialization}, best y: {np.min(yhist)}")
            # Get next sample point
            xsample = self.get_next_point()
            ysample = f(xsample)
            xhist = np.vstack((xhist, xsample))
            yhist = np.append(yhist, ysample)
            self.gaussian_process = GaussianProcess(xhist, yhist)
            self.gaussian_process.train()
            budget_minus_initialization -= 1

            if checkpoint_dir is not None and len(xhist) % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, seed, len(xhist), xhist, yhist)

        xhist, yhist = self.gaussian_process.get_historical_data()
        return xhist, -1 * yhist # save negative values as other experiments use *maximisation* of the negative objective

    def _save_checkpoint(self, checkpoint_dir, seed, iter_num, xhist, yhist):
        xs_path = os.path.join(checkpoint_dir, f"seed_{seed}_iter_{iter_num}_xs.npy")
        ys_path = os.path.join(checkpoint_dir, f"seed_{seed}_iter_{iter_num}_ys.npy")
        np.save(xs_path, xhist)
        np.save(ys_path, -1 * yhist)

    def get_next_point(self):
        # To be implemented by each acquisition function
        pass

    def save_bo_run(self, yhist, objective_name, seed):
        seed = str(seed)
        """
        Saves run to the folder ~/Look-Ahead/results/optimizer_name/objective_name/seed.csv
        """
        base_path = os.path.expanduser('~') + '/Look-Ahead/results/'
        # Make paths if necessary
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path + self.opt_name + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + objective_name + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        run_name = path + str(seed) + '.csv'

        # Save data as csv to path
        np.savetxt(run_name, yhist, delimiter=',')
