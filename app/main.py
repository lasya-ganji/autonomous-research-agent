from app.agent_runner import run_agent

if __name__ == "__main__":
    result = run_agent("Difference between CNN and RNN")
    print("\nFull Result:\n", result)
    print("\nReport:\n", result.get("report"))