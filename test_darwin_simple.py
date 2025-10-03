#!/usr/bin/env python3
"""
Teste Simplificado do Darwin Engine
===================================

Teste básico para verificar se o sistema funciona
sem dependências externas complexas.
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMNISTNet(nn.Module):
    """Rede neural simples para MNIST"""
    
    def __init__(self, hidden_size=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

class EvolvableMNIST:
    """MNIST Classifier evoluível - versão simplificada"""
    
    def __init__(self, genome=None):
        if genome is None:
            self.genome = {
                'hidden_size': random.choice([64, 128, 256]),
                'learning_rate': random.uniform(0.0001, 0.01),
                'batch_size': random.choice([32, 64, 128])
            }
        else:
            self.genome = genome
        
        self.fitness = 0.0
    
    def evaluate_fitness(self) -> float:
        """Avalia fitness com treino real"""
        try:
            # Criar modelo
            model = SimpleMNISTNet(self.genome['hidden_size'])
            
            # Carregar dados
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=self.genome['batch_size'], shuffle=True)
            
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
            
            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.genome['learning_rate'])
            
            # Treinar
            model.train()
            for epoch in range(3):  # Treino rápido
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    if batch_idx >= 100:  # Limitar para velocidade
                        break
            
            # Avaliar
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += len(data)
            
            accuracy = correct / total
            self.fitness = accuracy
            
            logger.info(f"   📊 Genome: {self.genome}")
            logger.info(f"   📊 Accuracy: {accuracy:.4f}")
            logger.info(f"   🎯 Fitness: {self.fitness:.4f}")
            
            return self.fitness
            
        except Exception as e:
            logger.error(f"   ❌ Fitness evaluation failed: {e}")
            self.fitness = 0.0
            return 0.0
    
    def mutate(self, mutation_rate=0.2):
        """Mutação genética"""
        new_genome = self.genome.copy()
        
        if random.random() < mutation_rate:
            key = random.choice(list(new_genome.keys()))
            
            if key == 'hidden_size':
                new_genome[key] = random.choice([64, 128, 256])
            elif key == 'learning_rate':
                new_genome[key] *= random.uniform(0.5, 2.0)
                new_genome[key] = max(0.0001, min(0.01, new_genome[key]))
            elif key == 'batch_size':
                new_genome[key] = random.choice([32, 64, 128])
        
        return EvolvableMNIST(new_genome)
    
    def crossover(self, other):
        """Crossover de ponto único"""
        child_genome = {}
        
        keys = list(self.genome.keys())
        crossover_point = random.randint(1, len(keys) - 1)
        
        for i, key in enumerate(keys):
            if i < crossover_point:
                child_genome[key] = self.genome[key]
            else:
                child_genome[key] = other.genome[key]
        
        return EvolvableMNIST(child_genome)

def test_evolution():
    """Teste de evolução simples"""
    logger.info("="*80)
    logger.info("🧪 TESTE SIMPLIFICADO DO DARWIN ENGINE")
    logger.info("="*80)
    
    # Teste 1: Fitness individual
    logger.info("\n🔬 Teste 1: Fitness Individual")
    individual = EvolvableMNIST()
    fitness = individual.evaluate_fitness()
    
    logger.info(f"✅ Fitness individual: {fitness:.4f}")
    
    # Teste 2: Múltiplos indivíduos
    logger.info("\n🔬 Teste 2: Múltiplos Indivíduos")
    fitnesses = []
    
    for i in range(3):
        logger.info(f"\n   Indivíduo {i+1}/3:")
        ind = EvolvableMNIST()
        fitness = ind.evaluate_fitness()
        fitnesses.append(fitness)
    
    logger.info(f"\n📊 Estatísticas:")
    logger.info(f"   Média: {np.mean(fitnesses):.4f}")
    logger.info(f"   Min: {np.min(fitnesses):.4f}")
    logger.info(f"   Max: {np.max(fitnesses):.4f}")
    logger.info(f"   Std: {np.std(fitnesses):.4f}")
    
    # Teste 3: Evolução simples
    logger.info("\n🔬 Teste 3: Evolução Simples (3 gerações)")
    
    population = [EvolvableMNIST() for _ in range(5)]
    
    for gen in range(3):
        logger.info(f"\n🧬 Geração {gen+1}/3:")
        
        # Avaliar
        for i, ind in enumerate(population):
            logger.info(f"   Indivíduo {i+1}/5:")
            ind.evaluate_fitness()
        
        # Ordenar
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        logger.info(f"\n   🏆 Melhor fitness: {population[0].fitness:.4f}")
        
        # Seleção + reprodução
        survivors = population[:3]  # Top 3
        
        offspring = []
        while len(survivors) + len(offspring) < 5:
            if random.random() < 0.8:
                p1, p2 = random.sample(survivors, 2)
                child = p1.crossover(p2).mutate()
            else:
                child = random.choice(survivors).mutate()
            offspring.append(child)
        
        population = survivors + offspring
    
    logger.info("\n" + "="*80)
    logger.info("✅ TESTE COMPLETO!")
    logger.info(f"   Melhor fitness final: {population[0].fitness:.4f}")
    logger.info(f"   Genoma: {population[0].genome}")
    logger.info("="*80)
    
    return population[0]

if __name__ == "__main__":
    best = test_evolution()