#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:58:32 2020

@author: b
https://www.youtube.com/watch?v=mSh4h-J0z1c&list=PLO_fdPEVlfKqMDNmCFzQISI2H_nJcEDJq&index=8
"""

# import d'un programme
#from 1a_type_de_donnees import table

import os
os.chdir ('/home/b/Documents/Python/Fichiers python')
os.listdir()


import math
import random
import statistics
import glob

##############################################################
# MATH
##############################################################
math.pi
math.cos(math.pi)


##############################################################
# STATISTICS
##############################################################
print(statistics.mean([1,2]))
print(statistics.variance([1,2]))



##############################################################
# RANDOM
##############################################################
essai = [1, 2, 3, 1]
print(random.choice(essai)) # Réalise un choix au hazard
random.seed(0) # choix de l'aélatoire
random.random()
random.randint(1, 10)
random.randrange(100)
random.sample(range(100), random.randrange(10)) # retourne un échantillon aléatoire
random.shuffle(essai) # mélange des données
essai

##############################################################
# OS
##############################################################
os.getcwd() # repertoire actuel de travail


##############################################################
# GLOB
##############################################################
glob.glob("*") # liste des fichiers présents
glob.glob("*.txt") # liste des fichiers présents

filenames = glob.glob("*txt")

for file in filenames:
    with open(file, 'r') as f:
        print(f.read())
        
# Lecture de toutes les lignes
with open('fichier.txt', 'r') as f:
    liste = f.readlines()
    
with open('fichier.txt', 'r') as f:
    liste = f.read().splitlines()
liste

# list comprehension
liste = [line.strip() for line in open('fichier.txt', 'r')]
liste


# Lecture de tous les fichiers et enregistrement
filenames = glob.glob("*txt")
d = {}
for file in filenames:
    with open(file, 'r') as f:
        d[file] = f.read().splitlines()
d