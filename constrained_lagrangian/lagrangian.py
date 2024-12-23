import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Constrained Lagrangian Optimization Model


def lagrangian_objective(x, obj_func, constraints, lambdas):
    """
    Compute the Lagrangian function.

    Parameters:
    - x: Decision variables.
    - obj_func: Objective function to minimize.
    - constraints: List of constraint functions.
    - lambdas: Lagrange multipliers.

    Returns:
    - Lagrangian value.
    """
    lagrangian = obj_func(x)
    for lambda_i, constraint in zip(lambdas, constraints):
        lagrangian += lambda_i * constraint(x)
    return lagrangian


def optimize_with_constraints(obj_func, constraints, x0, bounds, lambda_init):
    """
    Perform constrained optimization using Lagrangian multipliers.

    Parameters:
    - obj_func: Objective function to minimize.
    - constraints: List of constraint functions (equality constraints only).
    - x0: Initial guess for the decision variables.
    - bounds: Bounds for the decision variables.
    - lambda_init: Initial guess for Lagrange multipliers.

    Returns:
    - result: Optimization result.
    """
    n_constraints = len(constraints)

    def combined_objective(z):
        x = z[: len(x0)]
        lambdas = z[len(x0) :]
        return lagrangian_objective(x, obj_func, constraints, lambdas)

    z0 = np.concatenate(
        [x0, lambda_init]
    )  # Combine x0 and initial Lagrange multipliers

    # Define bounds for the decision variables and Lagrange multipliers
    combined_bounds = bounds + [(None, None)] * n_constraints

    # Solve optimization problem
    result = minimize(combined_objective, z0, bounds=combined_bounds, method="SLSQP")

    # Extract decision variables and Lagrange multipliers from result
    x_opt = result.x[: len(x0)]
    lambdas_opt = result.x[len(x0) :]

    return {
        "x_opt": x_opt,
        "lambdas_opt": lambdas_opt,
        "success": result.success,
        "message": result.message,
    }


# Example Usage
if __name__ == "__main__":
    # Define the objective function
    def objective(x):
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

    # Define the constraints (equality constraints: g(x) = 0)
    def constraint1(x):
        return x[0] + x[1] - 3

    def constraint2(x):
        return x[0] - 2

    constraints = [constraint1, constraint2]

    # Initial guess for decision variables and Lagrange multipliers
    x0 = np.array([0.5, 0.5])
    lambda_init = np.array([1.0, 1.0])

    # Bounds for the decision variables
    bounds = [(0, None), (0, None)]  # x >= 0, y >= 0

    # Solve the optimization problem
    result = optimize_with_constraints(objective, constraints, x0, bounds, lambda_init)

    # Print the results
    print("Optimal decision variables:", result["x_opt"])
    print("Optimal Lagrange multipliers:", result["lambdas_opt"])
    print("Optimization success:", result["success"])
    print("Message:", result["message"])

    x = np.linspace(0, 3, 100)
    y = 3 - x
    plt.plot(x, y, label="Constraint 1: x + y = 3")
    plt.axvline(x=2, color="r", linestyle="--", label="Constraint 2: x = 2")
    plt.scatter(
        result["x_opt"][0], result["x_opt"][1], color="g", label="Optimal Solution"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Constrained Optimization")
    plt.legend()
    plt.grid()
    plt.show()
