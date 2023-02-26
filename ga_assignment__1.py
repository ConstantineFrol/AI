

import math
import random
import string
import numpy as np

"""# Genetic Algorithm Assignment
30% of the overall grade for this module

Marks indciated in sections below are based on percentage of marks allocated for this module

In this assignment you must choose a problem, and attempt to use the Genetic Alogrithm that we developed in class to solve this problem.

## The Problem         **(~30%)**

*   Description of the problem

---

The problem is to use a genetic algorithm to guess a given string based on a scoring system. The genetic algorithm will start by creating a random population of strings of the same length as the target string. The fitness of each string in the population will be evaluated based on how close it is to the target string.

In this case, the scoring system will be based on dividing the length of a user input string into the number of characters and summing them together. If the total score equals 1, then the word is correct. For example, if the target string is "hello", and the input string is "hxllo", the score would be 4/5 or 0.8.

The genetic algorithm will use a selection process to choose the fittest individuals from the population to create the next generation. This process will include crossover and mutation to create new individuals with a combination of traits from the previous generation. The new generation will be evaluated for fitness, and the process will continue until a solution is found or a maximum number of generations is reached.

The goal is to find the string with a score of 1. The algorithm will track the best individuals in each generation and display them to the user. If the algorithm finds a string with a score of 1, it will stop and display the solution.

---

*   Discussion of the suitablity of Genetic Algorithms

---

In the given problem of a string guessing game, where the genetic algorithm is picking random characters and scoring them based on a formula that involves the length of the user input string, the suitability of genetic algorithms depends on the specific requirements of the problem and the available computational resources.

Genetic algorithms are well-suited for optimization problems, such as finding the best solution from a set of possible solutions. In this case, the genetic algorithm can be used to generate and evaluate possible solutions (strings) iteratively until a satisfactory result is achieved. However, genetic algorithms may not be the most efficient approach for this specific problem, as there are other techniques that can be used to solve it more quickly.

Additionally, the effectiveness of genetic algorithms depends on the quality of the fitness function used to evaluate the solutions. In this case, the fitness function is a simple formula based on the length of the user input string, which may not be able to capture the complexity of the problem. Therefore, it may be necessary to use a more sophisticated fitness function that takes into account other factors, such as the similarity between the guessed string and the actual target string.

Overall, the suitability of genetic algorithms for this specific problem depends on the specific requirements of the problem, the available computational resources, and the effectiveness of the fitness function used. It is important to carefully consider these factors before deciding whether to use genetic algorithms or other techniques to solve the problem.

One way to improve the algorithm's performance is to evaluate each candidate solution's fitness score based on how closely the string matches the target string. This can be done by counting the number of characters in the candidate solution that match the corresponding characters in the target string. The fitness score can then be normalized by dividing it by the length of the target string, giving a score between 0 and 1. Using this method can provide a more accurate measure of how well each candidate solution matches the target string, and can help the algorithm converge more quickly towards the optimal solution.

---

*   Complexity of the problem  (Overall marks allocated based on ..)

# The problem and the cost function   **(~20%)**
"""

class Problem:
    def __init__(self):
        self.target = "Hello world"

def fitness_function(chromosome, target):
    """returns the cost of each gene in chromosome"""
    # Validate if Char == Char of User input, then score +1
    score = 0.0
    for i in range(len(chromosome)):
        if chromosome[i] == target[i]:
            score += 1
    return score / len(target)

"""# The Individual **(~30%)**


*   Chromosone
*   Crossover
*   Mutation

## Discussion and justification on the approaches taken for the above

---

These two functions are used in the Individual class for generating a random character and selecting a random crossover point for crossover operation. 

The generate_character function uses random. the choice method from the random module to pick a random character from the set of ASCII letters and the space characters. 

The get_crossover_point function returns a random integer between 1 and the length of the chromosome minus 1, which is used to split the chromosomes of two-parent individuals during the crossover operation.

---
"""

def generate_character():
    """ Generate random char from alphabet	"""
    return random.choice(string.ascii_letters + " .")


def get_crossover_point(chromosome_length):
    """returns the random value in the range of incoming param"""
    return random.randint(1, chromosome_length - 1)

"""---

This is a 'Parameters' class that is used to store the parameters of the genetic algorithm.

The __init__ method initializes the parameters with default values. It takes an instance of the Problem class and uses the length of the target string to set the number of genes. It sets the mutation rate to 0.01, the population size to 200, and the output limit size to 10.

The purpose of this class is to provide a convenient way to store and access the parameters of the genetic algorithm so that they can be easily modified and passed around to different parts of the program.

---
"""

class Parameters:
    def __init__(self):
        problem = Problem()
        self.number_of_genes = len(problem.target)
        self.mutation_rate = 0.01
        self.population_size = 200
        self.output_limit_size = 10

"""---

This is the implementation of the Individual class which represents an individual in the population. It has properties and methods that are used to manipulate, evaluate and generate new individuals through mutation and crossover.

The __init__ method initializes an individual with a random chromosome of characters of length 'target_size'. The chromosome_to_string method returns the individual's chromosome as a string. The evaluate_fitness method calculates the cost of the initial solution by calling the 'fitness_function' method.

The 'crossover' method generates a new individual by combining two parent individuals' chromosomes at a random crossover point. The mutate method replaces some of the genes in the individual's chromosome with new randomly generated characters, depending on the mutation_rate.

Overall, the Individual class provides a framework for creating, evaluating, and manipulating individuals in the genetic algorithm population.

---
"""

# ######################################## Individual ######################################### #

class Individual(object):
    """ This Class representing an individual in the population. This class defines the properties and methods
    that are used to manipulate and evaluate the individual, as well as to generate new individuals
    through mutation and crossover"""

    def __init__(self, target_size):

        # Create array
        self.chromosome = []

        # Initialize a cost variable
        self.cost = 0.0

        # initialize array of characters
        for i in range(target_size):
            self.chromosome.append(generate_character())

    def chromosome_to_string(self):
        """returns chromosome as a string"""
        return ''.join(self.chromosome)

    def evaluate_fitness(self, target):
        """Calculate the cost of the initial solution"""
        self.cost = fitness_function(self.chromosome, target)

    def crossover(self, other_parent):
        """This is a method where child gain from parents genes"""
        child = Individual(len(self.chromosome))
        crossover_point = get_crossover_point(len(self.chromosome))
        child.chromosome = self.chromosome[0:crossover_point] + other_parent.chromosome[crossover_point:len(self.chromosome)]
        return child

    def mutate(self, mutation_rate):
        """Replacing old genes with new genes depends on the conditions"""
        # For each gene in chromosome
        for i in range(len(self.chromosome)):
            # Randomly picking genes from chromosome
            if random.random() < mutation_rate:
                # Replace gene with new random char
                self.chromosome[i] = generate_character()


# ##################################### Individual END ######################################### #

"""## Running the algorithm  **(~10%)**

*   Parameter choices
*   Modifications (if any) to run_genetic
*   Rationale for the above

---

The Population class is the main component of the implementation of the genetic algorithm. Its constructor takes as input a secret word, a mutation rate, and a population size. It initializes the population by creating a list of individual objects, where each individual has a chromosome of the same length as the secret word.

The calculate_fitness method is used to calculate the fitness score for each person in the population by calling the Assessment_fitness method of each Individual object.

The select_elite method selects the best people in the population based on their fitness score. It determines the fitness score of the fittest person and then adds people to the elite list based on their normalized fitness score.

The evolution method performs the basic operations of the genetic algorithm. It selects two parents from an elite list, crossovers to create a child, and then mutates it. This process is repeated until a new population of the same size as the previous population is created.

The get_fittest method returns the person with the highest fitness score.

The scoring method checks if the genetic algorithm has found a perfect match with the secret word. If so, it sets the completion flag to True.

The is_finished method checks if the genetic algorithm's stop condition is met. It returns a boolean indicating whether the algorithm has completed.

The get_num_generations method returns the number of generations created so far.

The calculate_average_fitness method calculates the average fitness score of all people in the population.

The get_phrases method returns a list of the best guesses generated by the genetic algorithm.

---
"""

class Population(object):
    """a class to represent a population of individuals in a genetic algorithm"""

    def __init__(self, secret_word, mutation_rate, population_size):

        # Initialize arrays
        self.population = []
        self.elite = []

        # Num of Generations
        self.generations = 0.0

        # Check finished or not, set to not
        self.finished = False

        # User input String
        self.target = secret_word

        # Mutation Rate
        self.rate_of_mutation = mutation_rate

        # Value(Cost)
        self.fitness_goal = 1

        # Empty String
        self.best_try = ""

        # Initialize Fitness Score
        self.fitness_score = None

        # Stuff population with individuals
        for _ in range(population_size):
            self.population.append(Individual(len(self.target)))

        # Valuate Each Gene
        self.calculate_fitness()

    def calculate_fitness(self):
        """this function calculates the cost of characters between the chromosome and the target string"""
        for i in range(len(self.population)):
            self.population[i].evaluate_fitness(self.target)

    def select_elite(self):
        """this method defines individual with the highest fitness_score"""

        self.elite = []
        self.fitness_score = 0.0

        for i in range(len(self.population)):
            if self.population[i].cost > self.fitness_score:
                self.fitness_score = self.population[i].cost

        for x in range(len(self.population)):
            fitness_cost = self.population[x].cost / self.fitness_score
            n = math.floor(fitness_cost * 100)
            for _ in range(n):
                self.elite.append(self.population[x])

    def evolve(self):
        """This method choose 2 parents, performs crossover to create a child, and mutates it"""
        for i in range(len(self.population)):
            parent1 = random.choice(self.elite)
            parent2 = random.choice(self.elite)
            child = parent1.crossover(parent2)
            child.mutate(self.rate_of_mutation)
            self.population[i] = child
        self.generations += 1

    def get_fittest(self):
        """returns the individual with the highest fitness score"""
        return self.best_try

    def evaluate(self):
        """if a perfect match is found, it sets the 'finished' flag to 'True'"""
        highest_fitness_score = 0.0
        index = 0
        for i in range(len(self.population)):
            if self.population[i].cost > highest_fitness_score:
                index = i
                highest_fitness_score = self.population[i].cost

        self.best_try = self.population[index].chromosome_to_string()
        if highest_fitness_score == self.fitness_goal:
            self.finished = True

    def is_finished(self):
        """
        method that checks if the stopping condition for the genetic algorithm is met,
        returns boolean True / False
        """
        return self.finished

    def get_num_generations(self):
        """returns the value of the generations"""
        return self.generations

    def calculate_average_fitness(self):
        """this functon calculates the average fitness score of all the individuals in th population"""
        total_fitness = 0
        for i in range(len(self.population)):
            total_fitness += self.population[i].cost

        return total_fitness / len(self.population)

    def get_phrases(self):
        """this method returns the best guesses produced by the genetic algorithm"""
        guesses = ""
        max_display = min(len(self.population), Parameters().output_limit_size)

        for i in range(max_display):
            guesses += self.population[i].chromosome_to_string() + "\n"

        return guesses

param = Parameters()
    problem = Problem()
    population = Population(problem.target, param.mutation_rate, param.population_size)

    while True:
        population.select_elite()
        population.evolve()
        population.calculate_fitness()
        population.evaluate()

        print(f"Phrases:\n{population.get_phrases()}")
        print(f"Generation No.:\t{str(population.get_num_generations())}")
        print(f"Best Solution:\t{population.get_fittest()}\n")

        if population.is_finished():
            break

    print("#################### Result ####################")
    print(f"Total generations\t\t{str(population.get_num_generations())}")
    print(f"Average Fitness\t\t\t{round(population.calculate_average_fitness(), 4)}")
    print(f"Total Population\t\t{str(param.population_size)}")
    print(f"Mutation Rate\t\t\t{str(param.mutation_rate * 100)}%")
    print(f"Your secret word was:\t\t{population.get_fittest()}")
    print("################################################")

#  If changes to params or reruns of iterations dont overwrite, create more cells and copy code down to show evolution of final solution

"""## Results and conclusions    **(~10%)**

---

After implementing and testing the genetic algorithm, it was able to generate candidate solutions that increasingly matched the target string, eventually arriving at a solution that matched the target string perfectly.

The algorithm initially generated a population of random individuals (strings), with each individual representing a possible solution to the problem. The fitness of each individual was calculated by counting the number of characters in the candidate solution that match the corresponding characters in the target string. The algorithm then selected the fittest individuals, performed crossover to create a child, and mutated it to introduce diversity in the population. After many generations, the algorithm was able to generate a solution that perfectly matched the target string.

The success of the algorithm shows the potential of genetic algorithms in solving problems that involve searching through a large number of possible solutions. However, the algorithm's success also depends on the problem's characteristics, such as the size of the solution space and the fitness function's effectiveness in evaluating candidate solutions.

---

---

EXAMPLE 2

---

In this sample, I conducted an experiment to compare the results obtained by using two different cost functions. The first function, fitness_function(), calculates the score of each correctly guessed character and sums up all the scores, which is then divided by the length of the target string to get the average score per character. On the other hand, the second function, calculate_cost(), assigns a numerical value to the position of each character in the chromosome and target string and calculates the difference between them. The sum of all the differences yields a single value, and the closer this value is to zero, the better the result.

Based on the experiment, I observed that the second function did not always yield better results than the first function. This could be because the characters generated randomly in each iteration of the experiment could be closer to the target string characters in either function, resulting in a better score.

Therefore, my conclusion is that it is challenging to determine which cost function would yield the best results since the characters are generated randomly. However, logically speaking, the second function should provide better results since it evaluates the proximity of each character to the target.
"""

import random
import string

import numpy as np


class Problem:
    def __init__(self):
        self.target = "Hello world"


class Parameters:
    def __init__(self):
        problem = Problem()
        self.number_of_genes = len(problem.target)
        self.mutation_rate = 0.09
        self.population_size = 200
        self.output_limit_size = 10


def generate_character():
    """ Generate random char from alphabet	"""
    return random.choice(string.ascii_letters + " .")


def get_crossover_point(chromosome_length):
    """returns the random value in the range of incoming param"""
    return random.randint(1, chromosome_length - 1)


def calculate_cost(chromosome, target_string):
    """
    Calculates the cost of each gene in the chromosome based on how far
    it is from the corresponding character in the target string.

    """
    cost = 0.0

    for i in range(len(chromosome)):
        cost += abs(ord(chromosome[i]) - ord(target_string[i]))

    return cost
    # return sum([1 if chromosome[i] != target_string[i] else 0 for i in range(len(target_string))])


class Individual(object):
    """ This Class representing an individual in the population. This class defines the properties and methods
    that are used to manipulate and evaluate the individual, as well as to generate new individuals
    through mutation and crossover"""

    def __init__(self, target_size):

        # Create array
        self.chromosome = []

        # Initialize a cost variable
        self.cost = 0

        # initialize array of characters
        for i in range(target_size):
            self.chromosome.append(generate_character())

    def chromosome_to_string(self):
        """returns chromosome as a string"""
        return ''.join(self.chromosome)

    def evaluate_fitness(self, target):
        """Calculate the cost of the initial solution"""
        self.cost = calculate_cost(self.chromosome, target)

    def crossover(self, other_parent):
        """This is a method where child gain from parents genes"""
        child = Individual(len(self.chromosome))
        crossover_point = get_crossover_point(len(self.chromosome))
        child.chromosome = self.chromosome[0:crossover_point] + other_parent.chromosome[crossover_point:len(self.chromosome)]
        return child

    def mutate(self, mutation_rate):
        """Replacing old genes with new genes depends on the conditions"""
        # For each gene in chromosome
        for i in range(len(self.chromosome)):
            # Randomly picking genes from chromosome
            if random.random() < mutation_rate:
                # Replace gene with new random char
                self.chromosome[i] = generate_character()


class Population(object):
    """a class to represent a population of individuals in a genetic algorithm"""

    def __init__(self, secret_word, mutation_rate, population_size):

        # Initialize arrays
        self.lowest_fitness_score = None
        self.population = []
        self.elite = []

        # Num of Generations
        self.generations = 0.0

        # Check finished or not, set to not
        self.finished = False

        # User input String
        self.target = secret_word

        # Mutation Rate
        self.rate_of_mutation = mutation_rate

        # Value(Cost)
        self.fitness_goal = 0

        # Empty String
        self.best_try = ""

        # Initialize Fitness Score
        self.fitness_score = np.infty

        # Stuff population with individuals
        for _ in range(population_size):
            self.population.append(Individual(len(self.target)))

        # Valuate Each Gene
        self.calculate_fitness()

    def calculate_fitness(self):
        """this function calculates the cost of characters between the chromosome and the target string"""
        for i in range(len(self.population)):
            self.population[i].evaluate_fitness(self.target)

    def select_elite(self):
        """this method defines individual with the highest fitness_score"""

        self.elite = []
        self.fitness_score = float('inf')

        for i in range(len(self.population)):
            if self.population[i].cost < self.fitness_score:
                self.fitness_score = self.population[i].cost
                self.elite.append(self.population[i])

    def evolve(self):
        """This method choose 2 parents, performs crossover to create a child, and mutates it"""
        for i in range(len(self.population)):
            parent1 = random.choice(self.elite)
            parent2 = random.choice(self.elite)
            child = parent1.crossover(parent2)
            child.mutate(self.rate_of_mutation)
            self.population[i] = child
        self.generations += 1


    def evaluate(self):
        """if a perfect match is found, it sets the 'finished' flag to 'True'"""
        index = 0
        # self.lowest_fitness_score = min(self.population, key=lambda x: x.cost).chromosome
        self.population.sort(key=lambda x: x.cost)
        lowest_fitness_score = self.population[index].cost

        self.best_try = self.population[index].chromosome_to_string()
        if lowest_fitness_score == self.fitness_goal:
            self.finished = True
        else:
            self.finished = False

    def get_fittest(self):
        """returns the individual with the highest fitness score"""
        return self.best_try

    def is_finished(self):
        """
        method that checks if the stopping condition for the genetic algorithm is met,
        returns boolean True / False
        """
        return self.finished

    def get_num_generations(self):
        """returns the value of the generations"""
        return self.generations

    def calculate_average_fitness(self):
        """this functon calculates the average fitness score of all the individuals in th population"""
        total_fitness = 0
        for i in range(len(self.population)):
            total_fitness += self.population[i].cost

        return total_fitness / len(self.population)

    def get_phrases(self):
        """this method returns the best guesses produced by the genetic algorithm"""
        guesses = ""
        max_display = min(len(self.population), Parameters().output_limit_size)

        for i in range(max_display):
            guesses += self.population[i].chromosome_to_string() + "\n"

        return guesses


def the_main():
    param = Parameters()
    problem = Problem()
    population = Population(problem.target, param.mutation_rate, param.population_size)

    while True:
        population.select_elite()
        population.evolve()
        population.calculate_fitness()
        population.evaluate()

        print(f"Generation No.:\t{str(population.get_num_generations())}")
        print(f"Best Solution:\t{population.get_fittest()}\n")

        if population.is_finished():
            break

    print("#################### Result ####################")
    print(f"Total generations\t\t{str(population.get_num_generations())}")
    print(f"Average Fitness\t\t\t{round(population.calculate_average_fitness(), 4)}")
    print(f"Total Population\t\t{str(param.population_size)}")
    print(f"Mutation Rate\t\t\t{str(param.mutation_rate * 100)}%")
    print(f"Your secret word was:\t{population.get_fittest()}")
    print("################################################")


the_main()

"""---

Individual Class

Function crossover():

This is a method that performs crossover between two parent individuals to create a new child individually.

First, a new Individual object is created with the same length as the parent's chromosomes. Then, a crossover point is determined using the get_crossover_point function.

Next, a new child chromosome is created by concatenating the first part of the current individual's chromosome up to the crossover point, and the second part of the other parent's chromosome starting from the crossover point up to the end of the chromosome.

Finally, the new child individual is returned with the newly created chromosome.

Overall, this method plays an important role in the evolution process by generating new individuals with a combination of characteristics from their parent individuals.

Function mutate():

This is a method implements mutation. The method takes a mutation_rate parameter which determines the probability of each gene in the chromosome being mutated.

The method iterates through each gene in the chromosome and randomly selects it with a probability of mutation_rate. If the gene is selected, it is replaced with a new randomly generated character.

Overall, this method implements a basic mutation mechanism that randomly alters the genetic material of an individual with a given probability.

---

Population Class

Function calculate_fitness():

The calculate_fitness() method calculates the fitness of each individual in the population by calling the evaluate_fitness() method of each individual and passing in the target string.

The evaluate_fitness() method calculates the cost of an individual's chromosome by comparing it to the target string. The lower the cost, the more fit the individual is. Therefore, calculate_fitness() is an important method in determining the fitness of each individual in the population, which is needed for selection and reproduction in the genetic algorithm.

Function select_elete():

The select_elite method selects the individuals with the best fitness scores and stores them in the elite list.

First, it initializes the elite list and sets the fitness_score to a very high value using float('inf'). Then, it loops through the entire population and compares each individual's cost with the current fitness_score. If an individual has a lower cost than the current fitness_score, it updates the fitness_score to that individual's cost and adds the individual to the elite list.

At the end of the loop, the elite list contains the individuals with the lowest cost values in the population. These individuals are then used in the evolve method to perform the selection, crossover, and mutation steps.

Function evolve():

The evolve() method is used to generate a new population of individuals by performing the genetic operations of selection, crossover, and mutation. The method selects a pair of parents from the elite group (the fittest individuals), creates a child by performing a crossover operation between the parents, and then mutates the child. This process is repeated for all individuals in the population.

The method has the following steps:

It loops through all individuals in the population and for each one, selects two parents from the elite group.
It creates a child by performing a crossover operation between the two parents.
It mutates the child using the mutation rate specified.
It replaces the current individual with the new child.
It increments the generation counter.
Overall, this method implements a basic genetic algorithm that iteratively improves the population over multiple generations until a satisfactory solution is found.

Function evaluate():

The evaluate method is used to evaluate the fitness of the individuals in the population and determine whether the optimization process is finished or not.

The method starts by setting the index to 0 and sorting the population based on the cost (fitness score) of each individual. The lowest fitness score is then obtained from the first individual in the sorted population, and the chromosome (solution) of this individual is saved as the best solution so far.

The method then checks if the lowest fitness score is equal to the fitness goal that was set for the optimization process. If it is, the finished attribute is set to True, indicating that the optimization process is complete. Otherwise, the finished attribute is set to False, indicating that the process needs to continue.

Overall, this method is a standard way to evaluate the fitness of individuals in an evolutionary algorithm and determine if the optimization process has converged to a satisfactory solution.

---

conclusions

---

In addition to my experiment, I discovered an interesting feature while running this code with different parameters and versions. Increasing the maximum population size resulted in a significant increase in the result. To illustrate this, let's take the example of a short target string "Hello" in the Problem class and increase the value of population_size to 1000 in the Parameters class.

When generating the chromosome, random characters are selected. In this case, we only need four characters. With 26 letters in the English alphabet, the chance of generating the required result that is close to the target is much higher. Mathematically, the population_size should be equal to 'pow(26, len(target))'. As the population size increases, the probability of generating the correct chromosome increases, resulting in fewer iterations, mutations, and crossovers.

This breaks the power of the genetic algorithm, which is based on the principle of repeating processes until the best solution is found. Therefore, increasing the population size too much may lead to suboptimal results, and careful consideration is required when choosing the appropriate population size.
"""

import random
import string

import numpy as np


class Problem:
    def __init__(self):
        self.target = "Hello"


class Parameters:
    def __init__(self):
        problem = Problem()
        self.number_of_genes = len(problem.target)
        self.mutation_rate = 0.09
        self.population_size = 1000
        self.output_limit_size = 10


def generate_character():
    """ Generate random char from alphabet	"""
    return random.choice(string.ascii_letters + " .")


def get_crossover_point(chromosome_length):
    """returns the random value in the range of incoming param"""
    return random.randint(1, chromosome_length - 1)


def calculate_cost(chromosome, target_string):
    """
    Calculates the cost of each gene in the chromosome based on how far
    it is from the corresponding character in the target string.

    """
    cost = 0.0

    for i in range(len(chromosome)):
        cost += abs(ord(chromosome[i]) - ord(target_string[i]))

    return cost
    # return sum([1 if chromosome[i] != target_string[i] else 0 for i in range(len(target_string))])


class Individual(object):
    """ This Class representing an individual in the population. This class defines the properties and methods
    that are used to manipulate and evaluate the individual, as well as to generate new individuals
    through mutation and crossover"""

    def __init__(self, target_size):

        # Create array
        self.chromosome = []

        # Initialize a cost variable
        self.cost = 0

        # initialize array of characters
        for i in range(target_size):
            self.chromosome.append(generate_character())

    def chromosome_to_string(self):
        """returns chromosome as a string"""
        return ''.join(self.chromosome)

    def evaluate_fitness(self, target):
        """Calculate the cost of the initial solution"""
        self.cost = calculate_cost(self.chromosome, target)

    def crossover(self, other_parent):
        """This is a method where child gain from parents genes"""
        child = Individual(len(self.chromosome))
        crossover_point = get_crossover_point(len(self.chromosome))
        child.chromosome = self.chromosome[0:crossover_point] + other_parent.chromosome[crossover_point:len(self.chromosome)]
        return child

    def mutate(self, mutation_rate):
        """Replacing old genes with new genes depends on the conditions"""
        # For each gene in chromosome
        for i in range(len(self.chromosome)):
            # Randomly picking genes from chromosome
            if random.random() < mutation_rate:
                # Replace gene with new random char
                self.chromosome[i] = generate_character()


class Population(object):
    """a class to represent a population of individuals in a genetic algorithm"""

    def __init__(self, secret_word, mutation_rate, population_size):

        # Initialize arrays
        self.lowest_fitness_score = None
        self.population = []
        self.elite = []

        # Num of Generations
        self.generations = 0.0

        # Check finished or not, set to not
        self.finished = False

        # User input String
        self.target = secret_word

        # Mutation Rate
        self.rate_of_mutation = mutation_rate

        # Value(Cost)
        self.fitness_goal = 0

        # Empty String
        self.best_try = ""

        # Initialize Fitness Score
        self.fitness_score = np.infty

        # Stuff population with individuals
        for _ in range(population_size):
            self.population.append(Individual(len(self.target)))

        # Valuate Each Gene
        self.calculate_fitness()

    def calculate_fitness(self):
        """this function calculates the cost of characters between the chromosome and the target string"""
        for i in range(len(self.population)):
            self.population[i].evaluate_fitness(self.target)

    def select_elite(self):
        """this method defines individual with the highest fitness_score"""

        self.elite = []
        self.fitness_score = float('inf')

        for i in range(len(self.population)):
            if self.population[i].cost < self.fitness_score:
                self.fitness_score = self.population[i].cost
                self.elite.append(self.population[i])

    def evolve(self):
        """This method choose 2 parents, performs crossover to create a child, and mutates it"""
        for i in range(len(self.population)):
            parent1 = random.choice(self.elite)
            parent2 = random.choice(self.elite)
            child = parent1.crossover(parent2)
            child.mutate(self.rate_of_mutation)
            self.population[i] = child
        self.generations += 1


    def evaluate(self):
        """if a perfect match is found, it sets the 'finished' flag to 'True'"""
        index = 0
        # self.lowest_fitness_score = min(self.population, key=lambda x: x.cost).chromosome
        self.population.sort(key=lambda x: x.cost)
        lowest_fitness_score = self.population[index].cost

        self.best_try = self.population[index].chromosome_to_string()
        if lowest_fitness_score == self.fitness_goal:
            self.finished = True
        else:
            self.finished = False

    def get_fittest(self):
        """returns the individual with the highest fitness score"""
        return self.best_try

    def is_finished(self):
        """
        method that checks if the stopping condition for the genetic algorithm is met,
        returns boolean True / False
        """
        return self.finished

    def get_num_generations(self):
        """returns the value of the generations"""
        return self.generations

    def calculate_average_fitness(self):
        """this functon calculates the average fitness score of all the individuals in th population"""
        total_fitness = 0
        for i in range(len(self.population)):
            total_fitness += self.population[i].cost

        return total_fitness / len(self.population)

    def get_phrases(self):
        """this method returns the best guesses produced by the genetic algorithm"""
        guesses = ""
        max_display = min(len(self.population), Parameters().output_limit_size)

        for i in range(max_display):
            guesses += self.population[i].chromosome_to_string() + "\n"

        return guesses


def the_main():
    param = Parameters()
    problem = Problem()
    population = Population(problem.target, param.mutation_rate, param.population_size)

    while True:
        population.select_elite()
        population.evolve()
        population.calculate_fitness()
        population.evaluate()

        print(f"Generation No.:\t{str(population.get_num_generations())}")
        print(f"Best Solution:\t{population.get_fittest()}\n")

        if population.is_finished():
            break

    print("#################### Result ####################")
    print(f"Total generations\t\t{str(population.get_num_generations())}")
    print(f"Average Fitness\t\t\t{round(population.calculate_average_fitness(), 4)}")
    print(f"Total Population\t\t{str(param.population_size)}")
    print(f"Mutation Rate\t\t\t{str(param.mutation_rate * 100)}%")
    print(f"Your secret word was:\t\t{population.get_fittest()}")
    print("################################################")


the_main()
