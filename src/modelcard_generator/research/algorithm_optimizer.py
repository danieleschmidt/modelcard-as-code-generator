"""Advanced algorithm optimization for model card generation."""

import asyncio
import hashlib
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core.logging_config import get_logger
from ..core.models import ModelCard, PerformanceMetric

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result from an optimization experiment."""
    parameters: Dict[str, Any]
    score: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentConfig:
    """Configuration for optimization experiments."""
    name: str
    parameter_space: Dict[str, Any]
    objective_function: str
    optimization_strategy: str = "bayesian"
    max_iterations: int = 100
    tolerance: float = 1e-6
    parallel_evaluations: int = 4
    random_seed: Optional[int] = None


class AlgorithmOptimizer:
    """Advanced optimization algorithms for model card generation and analysis."""

    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.evaluation_cache: Dict[str, OptimizationResult] = {}
        self.experiment_history: List[OptimizationResult] = []
        self.best_known_solutions: Dict[str, OptimizationResult] = {}
        
    def optimize_model_card_generation(
        self, 
        model_card: ModelCard,
        objective_function: Callable[[ModelCard, Dict[str, Any]], float],
        parameter_space: Dict[str, Any],
        strategy: str = "adaptive_differential_evolution"
    ) -> OptimizationResult:
        """Optimize model card generation parameters using advanced algorithms."""
        
        try:
            logger.info(f"Starting optimization for {model_card.model_details.name}")
            
            if strategy == "adaptive_differential_evolution":
                return self._adaptive_differential_evolution(
                    model_card, objective_function, parameter_space
                )
            elif strategy == "particle_swarm":
                return self._particle_swarm_optimization(
                    model_card, objective_function, parameter_space
                )
            elif strategy == "simulated_annealing":
                return self._simulated_annealing(
                    model_card, objective_function, parameter_space
                )
            elif strategy == "genetic_algorithm":
                return self._genetic_algorithm(
                    model_card, objective_function, parameter_space
                )
            else:
                raise ValueError(f"Unknown optimization strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

    def _adaptive_differential_evolution(
        self,
        model_card: ModelCard,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        max_generations: int = 100,
        population_size: int = 50
    ) -> OptimizationResult:
        """Adaptive Differential Evolution with self-adjusting parameters."""
        
        start_time = datetime.now()
        
        # Initialize population
        population = self._initialize_population(parameter_space, population_size)
        fitness_values = []
        
        # Adaptive parameters
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability
        adaptation_history = []
        
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(max_generations):
            generation_fitness = []
            
            for i, individual in enumerate(population):
                try:
                    # Create model card variant
                    card_variant = self._apply_parameters(model_card, individual)
                    
                    # Evaluate fitness
                    fitness = objective_function(card_variant, individual)
                    generation_fitness.append(fitness)
                    
                    # Update best solution
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_solution = individual.copy()
                    
                except Exception as e:
                    logger.warning(f"Evaluation failed for individual {i}: {e}")
                    generation_fitness.append(float('-inf'))
            
            fitness_values = generation_fitness
            
            # Adaptive parameter adjustment
            success_rate = self._calculate_success_rate(adaptation_history)
            if success_rate < 0.2:
                F = min(0.9, F * 1.1)  # Increase exploration
                CR = max(0.1, CR * 0.9)  # Decrease exploitation
            elif success_rate > 0.8:
                F = max(0.1, F * 0.9)  # Decrease exploration
                CR = min(0.9, CR * 1.1)  # Increase exploitation
            
            # Generate new population
            new_population = []
            successful_mutations = 0
            
            for i in range(population_size):
                # Select three random individuals (different from current)
                candidates = list(range(population_size))
                candidates.remove(i)
                a, b, c = random.sample(candidates, 3)
                
                # Differential mutation
                mutant = self._differential_mutation(
                    population[a], population[b], population[c], F, parameter_space
                )
                
                # Crossover
                trial = self._crossover(population[i], mutant, CR, parameter_space)
                
                # Selection
                trial_card = self._apply_parameters(model_card, trial)
                trial_fitness = objective_function(trial_card, trial)
                
                if trial_fitness > fitness_values[i]:
                    new_population.append(trial)
                    successful_mutations += 1
                else:
                    new_population.append(population[i])
            
            population = new_population
            adaptation_history.append(successful_mutations / population_size)
            
            # Keep only recent history
            if len(adaptation_history) > 10:
                adaptation_history.pop(0)
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}, F = {F:.3f}, CR = {CR:.3f}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            parameters=best_solution,
            score=best_fitness,
            execution_time=execution_time,
            metadata={
                "algorithm": "adaptive_differential_evolution",
                "generations": max_generations,
                "population_size": population_size,
                "final_F": F,
                "final_CR": CR
            }
        )
        
        self.experiment_history.append(result)
        logger.info(f"Optimization completed. Best score: {best_fitness:.4f}")
        
        return result

    def _particle_swarm_optimization(
        self,
        model_card: ModelCard,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        max_iterations: int = 100,
        swarm_size: int = 30
    ) -> OptimizationResult:
        """Particle Swarm Optimization with adaptive inertia."""
        
        start_time = datetime.now()
        
        # Initialize swarm
        particles = []
        velocities = []
        personal_bests = []
        personal_best_scores = []
        
        # Initialize particles
        for _ in range(swarm_size):
            particle = self._generate_random_parameters(parameter_space)
            velocity = self._generate_zero_velocity(parameter_space)
            
            particles.append(particle)
            velocities.append(velocity)
            personal_bests.append(particle.copy())
            
            # Evaluate initial fitness
            card_variant = self._apply_parameters(model_card, particle)
            score = objective_function(card_variant, particle)
            personal_best_scores.append(score)
        
        # Find global best
        global_best_idx = max(range(swarm_size), key=lambda i: personal_best_scores[i])
        global_best = personal_bests[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        # PSO parameters
        w_max = 0.9  # Maximum inertia weight
        w_min = 0.4  # Minimum inertia weight
        c1 = 2.0  # Cognitive coefficient
        c2 = 2.0  # Social coefficient
        
        for iteration in range(max_iterations):
            # Adaptive inertia weight
            w = w_max - (w_max - w_min) * iteration / max_iterations
            
            for i in range(swarm_size):
                # Update velocity
                r1, r2 = random.random(), random.random()
                
                for param in parameter_space:
                    if param in velocities[i]:
                        cognitive = c1 * r1 * (personal_bests[i][param] - particles[i][param])
                        social = c2 * r2 * (global_best[param] - particles[i][param])
                        
                        velocities[i][param] = (
                            w * velocities[i][param] + cognitive + social
                        )
                        
                        # Update position
                        particles[i][param] += velocities[i][param]
                        
                        # Apply bounds
                        particles[i] = self._apply_bounds(particles[i], parameter_space)
                
                # Evaluate fitness
                card_variant = self._apply_parameters(model_card, particles[i])
                current_score = objective_function(card_variant, particles[i])
                
                # Update personal best
                if current_score > personal_best_scores[i]:
                    personal_bests[i] = particles[i].copy()
                    personal_best_scores[i] = current_score
                
                # Update global best
                if current_score > global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = current_score
            
            if iteration % 10 == 0:
                logger.info(f"PSO Iteration {iteration}: Best score = {global_best_score:.4f}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            parameters=global_best,
            score=global_best_score,
            execution_time=execution_time,
            metadata={
                "algorithm": "particle_swarm_optimization",
                "iterations": max_iterations,
                "swarm_size": swarm_size,
                "final_inertia": w
            }
        )
        
        self.experiment_history.append(result)
        logger.info(f"PSO completed. Best score: {global_best_score:.4f}")
        
        return result

    def _simulated_annealing(
        self,
        model_card: ModelCard,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        max_iterations: int = 1000,
        initial_temperature: float = 100.0
    ) -> OptimizationResult:
        """Simulated Annealing with adaptive temperature schedule."""
        
        start_time = datetime.now()
        
        # Initialize
        current_solution = self._generate_random_parameters(parameter_space)
        current_card = self._apply_parameters(model_card, current_solution)
        current_score = objective_function(current_card, current_solution)
        
        best_solution = current_solution.copy()
        best_score = current_score
        
        temperature = initial_temperature
        cooling_rate = 0.95
        
        accepted_moves = 0
        total_moves = 0
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_solution, parameter_space)
            neighbor_card = self._apply_parameters(model_card, neighbor)
            neighbor_score = objective_function(neighbor_card, neighbor)
            
            # Calculate acceptance probability
            delta = neighbor_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_solution = neighbor
                current_score = neighbor_score
                accepted_moves += 1
                
                # Update best solution
                if current_score > best_score:
                    best_solution = current_solution.copy()
                    best_score = current_score
            
            total_moves += 1
            
            # Adaptive temperature schedule
            if iteration % 100 == 0:
                acceptance_rate = accepted_moves / max(total_moves, 1)
                if acceptance_rate < 0.1:
                    cooling_rate = 0.99  # Slower cooling
                elif acceptance_rate > 0.5:
                    cooling_rate = 0.9   # Faster cooling
                else:
                    cooling_rate = 0.95  # Normal cooling
                
                accepted_moves = 0
                total_moves = 0
            
            temperature *= cooling_rate
            
            if iteration % 100 == 0:
                logger.info(f"SA Iteration {iteration}: Best score = {best_score:.4f}, Temp = {temperature:.2f}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            parameters=best_solution,
            score=best_score,
            execution_time=execution_time,
            metadata={
                "algorithm": "simulated_annealing",
                "iterations": max_iterations,
                "initial_temperature": initial_temperature,
                "final_temperature": temperature,
                "cooling_rate": cooling_rate
            }
        )
        
        self.experiment_history.append(result)
        logger.info(f"SA completed. Best score: {best_score:.4f}")
        
        return result

    def _genetic_algorithm(
        self,
        model_card: ModelCard,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        generations: int = 100,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ) -> OptimizationResult:
        """Genetic Algorithm with adaptive parameters."""
        
        start_time = datetime.now()
        
        # Initialize population
        population = self._initialize_population(parameter_space, population_size)
        
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    card_variant = self._apply_parameters(model_card, individual)
                    fitness = objective_function(card_variant, individual)
                    fitness_scores.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_solution = individual.copy()
                        
                except Exception as e:
                    logger.warning(f"Fitness evaluation failed: {e}")
                    fitness_scores.append(float('-inf'))
            
            # Selection (Tournament selection)
            selected = []
            tournament_size = 3
            
            for _ in range(population_size):
                tournament_indices = random.sample(range(population_size), tournament_size)
                winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
                selected.append(population[winner_idx])
            
            # Crossover and mutation
            new_population = []
            
            for i in range(0, population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % population_size]
                
                if random.random() < crossover_rate:
                    child1, child2 = self._genetic_crossover(parent1, parent2, parameter_space)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < mutation_rate:
                    child1 = self._genetic_mutation(child1, parameter_space)
                if random.random() < mutation_rate:
                    child2 = self._genetic_mutation(child2, parameter_space)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
            
            if generation % 10 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                logger.info(f"GA Generation {generation}: Best = {best_fitness:.4f}, Avg = {avg_fitness:.4f}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            parameters=best_solution,
            score=best_fitness,
            execution_time=execution_time,
            metadata={
                "algorithm": "genetic_algorithm",
                "generations": generations,
                "population_size": population_size,
                "mutation_rate": mutation_rate,
                "crossover_rate": crossover_rate
            }
        )
        
        self.experiment_history.append(result)
        logger.info(f"GA completed. Best score: {best_fitness:.4f}")
        
        return result

    def benchmark_algorithms(
        self,
        model_card: ModelCard,
        objective_function: Callable,
        parameter_space: Dict[str, Any],
        algorithms: List[str] = None,
        iterations: int = 50
    ) -> Dict[str, OptimizationResult]:
        """Benchmark multiple optimization algorithms."""
        
        if algorithms is None:
            algorithms = [
                "adaptive_differential_evolution",
                "particle_swarm",
                "simulated_annealing",
                "genetic_algorithm"
            ]
        
        results = {}
        
        for algorithm in algorithms:
            logger.info(f"Benchmarking {algorithm}")
            
            try:
                result = self.optimize_model_card_generation(
                    model_card, objective_function, parameter_space, algorithm
                )
                results[algorithm] = result
                
                logger.info(f"{algorithm}: Score = {result.score:.4f}, Time = {result.execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Benchmarking failed for {algorithm}: {e}")
                results[algorithm] = None
        
        return results

    def _initialize_population(self, parameter_space: Dict[str, Any], size: int) -> List[Dict[str, Any]]:
        """Initialize population for evolutionary algorithms."""
        population = []
        for _ in range(size):
            individual = self._generate_random_parameters(parameter_space)
            population.append(individual)
        return population

    def _generate_random_parameters(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random parameters within the specified space."""
        parameters = {}
        
        for param_name, param_config in parameter_space.items():
            if param_config["type"] == "float":
                value = random.uniform(param_config["min"], param_config["max"])
            elif param_config["type"] == "int":
                value = random.randint(param_config["min"], param_config["max"])
            elif param_config["type"] == "choice":
                value = random.choice(param_config["choices"])
            elif param_config["type"] == "bool":
                value = random.choice([True, False])
            else:
                raise ValueError(f"Unsupported parameter type: {param_config['type']}")
            
            parameters[param_name] = value
        
        return parameters

    def _apply_parameters(self, model_card: ModelCard, parameters: Dict[str, Any]) -> ModelCard:
        """Apply optimization parameters to create a model card variant."""
        # Create a copy of the model card
        card_variant = ModelCard(model_card.config)
        
        # Copy all attributes
        card_variant.model_details = model_card.model_details
        card_variant.intended_use = model_card.intended_use
        card_variant.training_details = model_card.training_details
        card_variant.evaluation_results = model_card.evaluation_results.copy()
        card_variant.ethical_considerations = model_card.ethical_considerations
        card_variant.limitations = model_card.limitations
        card_variant.metadata = model_card.metadata.copy()
        card_variant.custom_sections = model_card.custom_sections.copy()
        
        # Apply parameter-specific modifications
        if "description_length" in parameters:
            # Modify description length
            desc = card_variant.model_details.description or ""
            target_length = parameters["description_length"]
            if len(desc) > target_length:
                card_variant.model_details.description = desc[:target_length] + "..."
        
        if "include_detailed_metrics" in parameters and parameters["include_detailed_metrics"]:
            # Add synthetic detailed metrics
            card_variant.add_metric("detailed_accuracy", 0.95)
            card_variant.add_metric("detailed_precision", 0.92)
            card_variant.add_metric("detailed_recall", 0.89)
        
        return card_variant

    def _differential_mutation(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any], 
        c: Dict[str, Any],
        F: float,
        parameter_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform differential mutation."""
        mutant = {}
        
        for param in parameter_space:
            if param in a and param in b and param in c:
                if parameter_space[param]["type"] in ["float", "int"]:
                    mutant[param] = a[param] + F * (b[param] - c[param])
                    
                    # Apply bounds
                    if "min" in parameter_space[param]:
                        mutant[param] = max(mutant[param], parameter_space[param]["min"])
                    if "max" in parameter_space[param]:
                        mutant[param] = min(mutant[param], parameter_space[param]["max"])
                    
                    if parameter_space[param]["type"] == "int":
                        mutant[param] = int(round(mutant[param]))
                else:
                    # For non-numeric parameters, randomly choose
                    mutant[param] = random.choice([a[param], b[param], c[param]])
            else:
                mutant[param] = a.get(param, self._generate_random_parameters({param: parameter_space[param]})[param])
        
        return mutant

    def _crossover(
        self,
        target: Dict[str, Any],
        mutant: Dict[str, Any],
        CR: float,
        parameter_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform crossover between target and mutant."""
        trial = {}
        
        # Ensure at least one parameter is taken from mutant
        j_rand = random.choice(list(parameter_space.keys()))
        
        for param in parameter_space:
            if random.random() <= CR or param == j_rand:
                trial[param] = mutant.get(param, target.get(param))
            else:
                trial[param] = target.get(param)
        
        return trial

    def _calculate_success_rate(self, history: List[float]) -> float:
        """Calculate success rate from adaptation history."""
        if not history:
            return 0.5
        return sum(history) / len(history)

    def _generate_zero_velocity(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate zero velocity vector for PSO."""
        velocity = {}
        for param in parameter_space:
            if parameter_space[param]["type"] in ["float", "int"]:
                velocity[param] = 0.0
        return velocity

    def _apply_bounds(self, parameters: Dict[str, Any], parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter bounds."""
        bounded = parameters.copy()
        
        for param, config in parameter_space.items():
            if param in bounded and config["type"] in ["float", "int"]:
                if "min" in config:
                    bounded[param] = max(bounded[param], config["min"])
                if "max" in config:
                    bounded[param] = min(bounded[param], config["max"])
                
                if config["type"] == "int":
                    bounded[param] = int(round(bounded[param]))
        
        return bounded

    def _generate_neighbor(self, solution: Dict[str, Any], parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate neighbor solution for simulated annealing."""
        neighbor = solution.copy()
        
        # Randomly select parameter to modify
        param = random.choice(list(parameter_space.keys()))
        config = parameter_space[param]
        
        if config["type"] == "float":
            # Add Gaussian noise
            noise = random.gauss(0, (config["max"] - config["min"]) * 0.1)
            neighbor[param] = max(config["min"], min(config["max"], solution[param] + noise))
        elif config["type"] == "int":
            # Random walk
            change = random.choice([-1, 0, 1])
            neighbor[param] = max(config["min"], min(config["max"], solution[param] + change))
        elif config["type"] == "choice":
            neighbor[param] = random.choice(config["choices"])
        elif config["type"] == "bool":
            neighbor[param] = not solution[param]
        
        return neighbor

    def _genetic_crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        parameter_space: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform genetic crossover."""
        child1, child2 = {}, {}
        
        for param in parameter_space:
            if random.random() < 0.5:
                child1[param] = parent1.get(param)
                child2[param] = parent2.get(param)
            else:
                child1[param] = parent2.get(param)
                child2[param] = parent1.get(param)
        
        return child1, child2

    def _genetic_mutation(self, individual: Dict[str, Any], parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Perform genetic mutation."""
        mutated = individual.copy()
        
        for param, config in parameter_space.items():
            if random.random() < 0.1:  # 10% mutation rate per parameter
                if config["type"] == "float":
                    noise = random.gauss(0, (config["max"] - config["min"]) * 0.05)
                    mutated[param] = max(config["min"], min(config["max"], individual[param] + noise))
                elif config["type"] == "int":
                    mutated[param] = random.randint(config["min"], config["max"])
                elif config["type"] == "choice":
                    mutated[param] = random.choice(config["choices"])
                elif config["type"] == "bool":
                    mutated[param] = not individual[param]
        
        return mutated