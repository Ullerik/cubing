import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
import random

# 2x2 cube
solved_state = np.array([0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,0])

move_map = {
    "R": np.array([0, 2, 5, 3, 4, 6, 1, 7, 0, 1, 2, 0, 0, 1, 2, 0]),
    "U": np.array([3, 0, 1, 2, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0]),
    "F": np.array([0, 1, 3, 4, 5, 2, 6, 7, 0, 0, 1, 2, 1, 2, 0, 0]),
    "R'": np.array([0, 6, 1, 3, 4, 2, 5, 7, 0, 1, 2, 0, 0, 1, 2, 0]),
    "U'": np.array([1, 2, 3, 0, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0]),
    "F'": np.array([0, 1, 5, 2, 3, 4, 6, 7, 0, 0, 1, 2, 1, 2, 0, 0]),
    "R2": np.array([0, 5, 6, 3, 4, 1, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0]),
    "U2": np.array([2, 3, 0, 1, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0]),
    "F2": np.array([0, 1, 4, 5, 2, 3, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0]),
}

moves = ["U", "U2", "U'", "R", "R2", "R'", "F", "F2", "F'"]

@njit
def apply_move(cube, move_array):
    cube2 = cube.copy()
    # Orientation
    cube[8:16] = (cube2[(move_array[:8]+8)] + move_array[8:16]) % 3 

    # rewrite the orientation
    cube[:8] = cube[move_array[:8]]

    return cube

def apply_alg(cube, alg):
    for move in alg.split(" "):
        cube = apply_move(cube, move_map[move])

    return cube

def get_cube(scramble = ""):
    if scramble == "":
        return np.array([0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,0])
    else:
        cube = np.array([0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,0])
        cube = apply_alg(cube, scramble)
        return cube

cube = get_cube("")

move_array = move_map["R"]
cube = apply_move(cube, move_array)

def inv(solution):
    return " ".join(
        (s[:-1] if "'" in s else s + "'") if "2" not in s else s
        for s in solution.split()[::-1]
    )

# we wanna make a solver
# id 7 is never used

@njit
def get_id_from_state(cube):
    ID0 = 0
    for i in range(6):
        ID0 += cube[i] * 7**i
    
    ID1 = 0
    for i in range(6):
        ID1 += cube[i+8] * 3**(i)
    
    return ID0*3**6 + ID1

cube = get_cube("R U R' U' R' F R2 U' R' U' R U R' F'")
cube = get_cube(inv("F R2 U2 F U2 R F' R"))
get_id_from_state(cube)


@njit
def _inc(ids):
    for i in range(len(ids)):
        ids[i]+=1
        ids[i]%=9
        if ids[i]:
            break
    return ids

@njit
def _is_valid(length,ids):
    for i in range(length-1):
        if ids[i]//3==ids[i+1]//3:
            return False
    return True      

@njit
def _increment(length,ids):
    ids = _inc(ids)
    while not _is_valid(length,ids):
        ids = _inc(ids)
    return ids


class alg_index:
    def __init__(self, depth):
        self.depth = depth
        self.alg = np.array([0,3,6]*(depth//3+1))[:depth]

    def increment(self):
        self.alg = _increment(self.depth,self.alg)

    def __str__(self):
        return " ".join(moves[i] for i in self.alg)
    

ai = alg_index(3)

def gen_all_algs(depth, print_progress = False):
    all_algs = []
    for i in range(1, depth+1):
        if print_progress:
            print(f"Genning algs of length {i}...")
        ai = alg_index(i)
        start_alg = str(ai)
        ai.increment()
        while str(ai) != start_alg:
            all_algs.append(str(ai))
            ai.increment()

    return all_algs

def gen_table(depth, print_progress = False):
    algs = gen_all_algs(depth, print_progress)
    table = {}
    if print_progress:
        print("Generating table...")
    for alg in algs:
        cube = get_cube(alg)
        ID = get_id_from_state(cube)
        if ID not in table:
            table[ID] = inv(alg)

    return table

table = gen_table(8, True)

def solver(cube, search_algs, table):
    ID = get_id_from_state(cube)
    if ID in table:
        return table[ID]
    
    for alg in search_algs:
        cube = apply_alg(cube, alg)
        ID = get_id_from_state(cube)
        if ID in table:
            return alg + " " + table[ID]
        cube = apply_alg(cube, inv(alg))
        
    return "No solution found"

search_algs = gen_all_algs(3)
cube = get_cube("R U R' U' R' F R2 U' R' U' R U R' F'")
solver(cube, search_algs, table)