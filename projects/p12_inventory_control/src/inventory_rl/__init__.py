"""Inventory control with reinforcement learning (project 12).

Importing the package registers the environment with Gymnasium::

    import gymnasium as gym
    env = gym.make("inventory_rl/InventoryControl-v0")  # a demo product
    env = gym.make("inventory_rl/InventoryControl-v0", problem=Problem.from_product(product))
"""

import gymnasium as gym

from inventory_rl.dynamics import Problem

__all__ = ["Problem"]
__version__ = "0.1.0"

ENV_ID = "inventory_rl/InventoryControl-v0"

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="inventory_rl.env:InventoryEnv")
