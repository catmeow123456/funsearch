from funsearch.implementation import funsearch
from funsearch.implementation.code_manipulation_test import create_test_program

# program = create_test_program(has_imports=True, has_class=True, has_assignment=True)
# print(program)
program = '''\
"""Finds large cap sets."""
import itertools
import numpy as np
class funsearch:
    def run(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    def evolve(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

@funsearch.run
def evaluate(n: int) -> int:
  """Returns the size of an `n`-dimensional cap set."""
  capset = solve(n)
  return len(capset)


def solve(n: int) -> np.ndarray:
  """Returns a large cap set in `n` dimensions."""
  all_vectors = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

  # Powers in decreasing order for compatibility with `itertools.product`, so
  # that the relationship `i = all_vectors[i] @ powers` holds for all `i`.
  powers = 3 ** np.arange(n - 1, -1, -1)

  # Precompute all priorities.
  priorities = np.array([priority(tuple(vector), n) for vector in all_vectors])

  # Build `capset` greedily, using priorities for prioritization.
  capset = np.empty(shape=(0, n), dtype=np.int32)
  while np.any(priorities != -np.inf):
    # Add a vector with maximum priority to `capset`, and set priorities of
    # invalidated vectors to `-inf`, so that they never get selected.
    max_index = np.argmax(priorities)
    vector = all_vectors[None, max_index]  # [1, n]
    blocking = np.einsum('cn,n->c', (- capset - vector) % 3, powers)  # [C]
    priorities[blocking] = -np.inf
    priorities[max_index] = -np.inf
    capset = np.concatenate([capset, vector], axis=0)

  return capset


@funsearch.evolve
def priority(el: tuple[int, ...], n: int) -> float:
  """Returns the priority with which we want to add `element` to the cap set."""
  return 0.0
'''

# print(funsearch._extract_function_names(program))
funsearch.main(program,
               [3, 4, 5, 6, 7, 8, 9],
               funsearch.config_lib.Config())
