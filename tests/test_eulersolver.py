import os
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automated_systems.SimpleEulerSolver import build_system
from langchain_core.messages import HumanMessage

if __name__ == "__main__":

    # result = 4179871
    problem = """A perfect number is a number for which the sum of its proper divisors is exactly equal to the number. For example, the sum of the proper divisors of 28 would be 1 + 2 + 4 + 7 + 14 = 28, which means that 28 is a perfect number.

A number n is called deficient if the sum of its proper divisors is less than n and it is called abundant if this sum exceeds n.

As 12 is the smallest abundant number, 1 + 2 + 3 + 4 + 6 = 16, the smallest number that can be written as the sum of two abundant numbers is 24. By mathematical analysis, it can be shown that all integers greater than 28123 can be written as the sum of two abundant numbers. However, this upper limit cannot be reduced any further by analysis even though it is known that the greatest number that cannot be expressed as the sum of two abundant numbers is less than this limit.

Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers."""
    
    workflow, system = build_system()
    inputs = {"messages": [], "problem_description": problem}
    
    print("\nStreaming solution process...\n")
    print("-" * 60)

    for output in workflow.stream(inputs):
        for key, value in output.items():
            # Get the latest message
            latest_message = value['messages'][-1]
            print(f"\n[{latest_message.type} {key}]: {latest_message}")
            print("-" * 60)
        time.sleep(1)


