# tweak everything for a run from here.
# state index convention: row*n + col

n = 5
m = 5

formula = "(!c U b) & (F c) & (F a) & (!d U a) & (!d U c)"

regions = ['a', 'b', 'c', 'd']

# where each label actually lives. cells not listed have nothing.
# a cell can list multiple atoms if they all hold there.
true_locations = {
    22: ['a'],
    24: ['b'],
    20: ['c'],
    3: ['d'],
    16: ['d'],
    21: ['d'],
}

# initial belief: {cell: {label: prob}}.
# you only need to write the atoms that are TRUE in that label,
# the missing ones are taken as false. so for regions = ['a','b','c','d']:
#   'a'     means  a && !b && !c && !d
#   'a & b' means  a &&  b && !c && !d
#   ''      means !a && !b && !c && !d   (nothing holds)
# anything you don't put a number on goes to the "nothing holds" label,
# and the row is normalized to sum to 1.
# example: {'a': 0.3} on a cell -> 0.3 on a-only, 0.7 on nothing-holds.
# set initial_belief = None to skip per-cell priors (all cells default to nothing-holds = 1).
initial_belief = {
    2:{'a':0.5},
    4:{'a':0.8},
    22:{'a':0.4},
    9:{'b':0.4},
    24:{'b':0.4},
    17:{'b':0.1},
    10:{'c':0.4},
    18:{'c':0.9},
    20:{'c':0.7},
    13:{'d':0.2},
    16:{'d':0.5},
    }
