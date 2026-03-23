from app.agent_runner import run_agent

if __name__ == "__main__":
    result = run_agent("Impact of AI on healthcare")
    print("\nFull Result:\n", result)
    print("\nReport:\n", result.get("report"))