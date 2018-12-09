
# coding: utf-8

# In[1]:


from utils import *
from mcmc import *

class Experiment:
    def __init__(self, n, alpha, N, b_v, sampler=MCMC.metropolis, 
               show_plot=False, print_statistics=False):
        self.n = n
        self.alpha = alpha
        self.m = int(alpha * n)
        self.N = N
        self.sampler = sampler

        self.show_plot = show_plot
        self.print_statistics = print_statistics

        self.mcmc = MCMC(self.n, self.m, sampler=self.sampler)
        self.x_v = np.zeros((1, n))  # state vector
        self.h_v = np.zeros(1)       # energy vector
        self.a_v = np.zeros(1)       # acceptance probability vector
        self.s_v = np.zeros(1)       # acceptance result vector
        self.e_v = np.zeros(1)       # error vector
        self.b_v = b_v               # beta vector

    def run(self):
        raise NotImplementedError("To implement in subsclasses")
        
    def final_error(self):
        return self.e_v[-1]
    
    def final_energy(self):
        return self.h_v[-1]
    
    def min_energy(self):
        return self.h_v.min()
    
    def error_at_min_energy(self):
        return self.e_v[self.h_v.argmin()]
    
    def min_error(self):
        return self.e_v.min()
    
    def energy_at_min_error(self):
        return self.h_v[self.e_v.argmin()]
    
    def print_error_energy_statistics(self):
        
        '''print('%35s %.4f ' %('Energy Last Sample:', self.final_energy()))
        print('%35s %.4f ' %('Minimum Energy Sample:', self.min_energy()))
        print('%35s %.4f ' %('Energy for Minimum Error Sample:',
                           self.error_at_min_energy()))
        
        print('%35s %.4f ' %('Error Last Sample:', self.final_error()))
        print('%35s %.4f ' %('Error for Minimum Energy Sample:',
                           self.error_at_min_energy()))
        print('%35s %.4f ' %( 'Minimum Error Sample:', self.min_error()))
        print('')'''
        
        print('%35s %.4f, %35s %.4f ' %('Energy Last Sample:', self.final_energy(), 
                                        'Error Last Sample:', self.final_error()))
        print('%35s %.4f, %35s %.4f ' %('Minimum Energy Sample:', self.min_energy(), 
                                        'Error for Minimum Energy Sample:',
                                        self.error_at_min_energy()))
        print('%35s %.4f, %35s %.4f ' %('Energy for Minimum Error Sample:',
                                        self.energy_at_min_error(), 'Minimum Error Sample:',
                                        self.min_error()))
        print('')
        
        

    def plot_energy_error(self, title=None):
        if title is None:
            title = "n = " + self.n + ", alpha = " + self.alpha
        plot_energy_error([self.h_v], [self.e_v], title)
    
    def plot_acceptance_trend(self, title=None):
        if title is None:
            title = "n = " + self.n + ", alpha = " + self.alpha
        plot_acceptance_trend(self.a_v, self.s_v, title)

    def plot_beta_schedule(self):
        plot_beta_schedule(self.b_v)
    
    def plot_all_results(self, title=None):
        if title is None:
            title = "n = " + str(self.n) + ", alpha = " + str(self.alpha)
        self.plot_energy_error(title)
        self.plot_acceptance_trend("")
        self.plot_beta_schedule()

        
class StandardExperiment(Experiment):
    """The StandardExperiment stores all interesting data for all timesteps,
    this could take a lot of memory."""
    def __init__(self, n, alpha, N, b_v, sampler=MCMC.metropolis, 
               show_plot=False, print_statistics=False):
        super().__init__(n, alpha, N, b_v, sampler, show_plot, print_statistics)

        self.x_v = np.zeros((N + 1, n))  # state vector
        self.h_v = np.zeros(N + 1)       # energy vector
        self.a_v = np.zeros(N + 1)       # acceptance probability vector
        self.s_v = np.zeros(N + 1)       # acceptance result vector
        self.e_v = np.zeros(N + 1)       # error vector

    def run(self):
        self.x_v[0], self.h_v[0], self.a_v[0], self.s_v[0] = self.mcmc.get_initial_state()
        self.e_v[0] = self.mcmc.error(self.x_v[0])
        self.mcmc.set_beta(self.b_v[0])
        for i in range(1, self.N + 1):
            self.x_v[i], self.h_v[i], self.a_v[i], self.s_v[i] = self.mcmc.draw_sample(
                self.x_v[i-1])
            self.e_v[i] = self.mcmc.error(self.x_v[i])
            if i < self.N and self.b_v[i] != self.b_v[i-1]:
                self.mcmc.set_beta(self.b_v[i])
                #print( 'Setting beta to %.2f, %.2f' %(self.b_v[i], self.mcmc.beta))

        if self.show_plot:
            self.plot_all_results()

        if self.print_statistics:
            self.print_error_energy_statistics()

        return self
    
class LeanExperiment(Experiment):
    """The LeanExperiment stores all interesting data for all timesteps except the chain.
    this should save a lot of memory."""
    def __init__(self, n, alpha, N, b_v, sampler=MCMC.metropolis, 
               show_plot=False, print_statistics=False):
        super().__init__(n, alpha, N, b_v, sampler, show_plot, print_statistics)

        self.h_v = np.zeros(N + 1)       # energy vector
        self.a_v = np.zeros(N + 1)       # acceptance probability vector
        self.s_v = np.zeros(N + 1)       # acceptance result vector
        self.e_v = np.zeros(N + 1)       # error vector

    def run(self):
        self.x_v[0], self.h_v[0], self.a_v[0], self.s_v[0] = self.mcmc.get_initial_state()
        self.e_v[0] = self.mcmc.error(self.x_v[0])
        self.mcmc.set_beta(self.b_v[0])
        for i in range(1, self.N + 1):
            self.x_v[0], self.h_v[i], self.a_v[i], self.s_v[i] = self.mcmc.draw_sample(
                self.x_v[0])
            self.e_v[i] = self.mcmc.error(self.x_v[0])
            if i < self.N and self.b_v[i] != self.b_v[i-1]:
                self.mcmc.set_beta(self.b_v[i])
                #print( 'Setting beta to %.2f, %.2f' %(self.b_v[i], self.mcmc.beta))

        if self.show_plot:
            self.plot_all_results()

        if self.print_statistics:
            self.print_error_energy_statistics()

        return self
    

class Experiments:
    def __init__(self, n_vector, alpha_vector, N, b_v, sampler=MCMC.metropolis,
               show_plot=False, print_statistics=False, seed=123, lean=True):
        """Creates and runs MCMC experiments. <n> and <alpha> can be lists,
           experiments for all combinations of their parameters will be run.
           N and b_v should (currently) stay constant. The main seed is set during 
           initialization and unique seeds are generated for each MCMC experiment, 
           so it's also possible to run multiple times with the same parameters."""
        if seed != -1:
            self.set_seed(seed)


        
        self.parameters = np.array(
            np.meshgrid(n_vector, alpha_vector)).T.reshape(-1, 2)
        print("Running %d experiments." % len(self.parameters))
        
        #To ensure same data when comparing against different N
        self.seeds = np.random.randint(1000000, size=len(self.parameters))
        self.N = N
        self.b_v = b_v

        # Not 
        self.experiments = defaultdict(defaultdict)
        for indx, [n, alpha] in enumerate( self.parameters ):
            n = int(n)
            if lean:
                Exp = LeanExperiment
            else:
                Exp = StandardExperiment
            
            #To ensure same data when comparing against different N
            self.set_seed( self.seeds[indx] )
            self.experiments[indx] = Exp(n, alpha, N, b_v, sampler,
                                               show_plot, print_statistics).run()
            

    def set_seed(self, seed):
        np.random.seed(seed)
    
    def __getitem__(self, key):
        n, alpha = self.parameters[key]
        return self.experiments[key]
  
    def __len__(self):
        return len(self.parameters)
  
    class _exp_iter:
        def __init__(self, experiments, parameters):
            self.experiments = experiments
            self.parameters = parameters
            self.cur = 0

        def __next__(self):
            i = self.cur
            if i >= len(self.parameters):
                raise StopIteration
            self.cur += 1
            n, alpha = self.parameters[i]
            return self.experiments[i]

    def __iter__(self):
        return Experiments._exp_iter(self.experiments, self.parameters)
  
    def final_errors(self):
        e_v = np.ones(len(self.parameters))
        for i, experiment in enumerate(self):
            e_v[i] = experiment.final_error()
        return e_v
    
    def min_energy_errors(self):
        e_v = np.ones(len(self.parameters))
        for i, experiment in enumerate(self):
        
            e_v[i] = experiment.error_at_min_energy()
        return e_v
    
    def min_errors(self):
        e_v = np.ones(len(self.parameters))
        for i, experiment in enumerate(self):
            e_v[i] = experiment.min_error()
        return e_v
    
    def final_energies(self):
        h_v = np.ones(len(self.parameters))
        for i, experiment in enumerate(self):
            h_v[i] = experiment.final_energy()
        return h_v

    def min_energies(self):
        h_v = np.ones(len(self.parameters))
        for i, experiment in enumerate(self):
            h_v[i] = experiment.min_energy()
        return h_v
    
    def min_error_energies(self):
        h_v = np.ones(len(self.parameters))
        for i, experiment in enumerate(self):        
            h_v[i] = experiment.energy_at_min_error()
        return h_v
    
    
class MultipleExperiments:
    
    
    def __init__(self, n, alpha, N, b_v, n_exp=10, sampler=MCMC.metropolis,
               show_plot=False, print_statistics=False, seed=123, schedule_name='', lean=True):
        """Creates and runs n_exp MCMC experiments for a particular setting of
           n and alpha, N and b_v should (currently) stay constant. The seeds 
           for each of the n_exp experiment is set by the Experiments class, so
           so it's possible to compare different schedules (b_v) and N for same
           n and alpha by fixing the seed."""
        
        self.n_exp = n_exp
        self.alpha_vector = [alpha]
        self.n_vector = [n]*self.n_exp
        self.sampler_name = sampler.__name__
        self.schedule_name = schedule_name
        self.description = 'Running %d experiments with \n n:%d, alpha:%.2f, N:%d steps, Sampler:%s, \n Schedule:%s'                     %( n_exp, n, alpha, N, self.sampler_name, self.schedule_name)
        
        self.exp = Experiments(self.n_vector, self.alpha_vector, N, b_v, sampler, show_plot,
                      print_statistics, seed, lean)
    
    def get_mean_final_error(self):        
        return np.mean( self.exp.final_errors() )
    
    def get_mean_min_error( self ):
        return np.mean( self.exp.min_errors() )
    
    def get_mean_error_at_min_energy(self):
        return np.mean( self.exp.min_energy_errors() )

    def get_mean_final_energy(self):
        return np.mean( self.exp.final_energies() )
    
    def get_mean_energy_at_min_error(self):
        return np.mean( self.exp.min_error_energies() )

    def get_mean_min_energy(self):
        return np.mean( self.exp.min_energies() )
    
    def get_std_final_error(self):
        return np.std( self.exp.final_errors() )
    
    def get_std_min_error(self):
        return np.std( self.exp.min_errors() )
    
    def get_std_error_at_min_energy(self):
        return np.std( self.exp.min_energy_errors() )

    def get_std_final_energy(self):
        return np.std( self.exp.final_energies() )
    
    def get_std_energy_at_min_error(self):
        return np.std( self.exp.min_error_energies() )
    
    def get_std_min_energy(self):
        return np.std( self.exp.min_energies() )
    
    def print_mean_statistics( self ):
        print( self.description )
        print('\n')
        
        print( '%35s: %.3f, %35s: %.3f' %('Mean Final Energy', self.get_mean_final_energy(), 
                                          'Mean Final Error', self.get_mean_final_error()))
        
        print( '%35s: %.3f, %35s: %.3f' %('Mean Minimum Energy', self.get_mean_min_energy(), 
                                          'Mean Error at Minimum Energy', self.get_mean_error_at_min_energy()))
        
        print( '%35s: %.3f, %35s: %.3f' %('Mean Energy at Minimum Error', self.get_mean_energy_at_min_error(), 
                                          'Mean Minimum Error', self.get_mean_min_error()))

        print( '%35s: %.3f, %35s: %.3f' %('Std Final Energy', self.get_std_final_energy(), 
                                          'Std Final Error', self.get_std_final_error()))
        
        print( '%35s: %.3f, %35s: %.3f' %('Std Minimum Energy', self.get_std_min_energy(), 
                                          'Std Error at Minimum Energy', self.get_std_error_at_min_energy()))
        
        print( '%35s: %.3f, %35s: %.3f' %('Std Energy at Minimum Error', self.get_std_energy_at_min_error(), 
                                          'Std Minimum Error', self.get_std_min_error()))

