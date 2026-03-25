from app.agent_runner import run_agent

if __name__ == "__main__":
    result = run_agent("Difference between CNN and RNN")
    print("\nFull Result:\n", result)
    report = result.get("report")
    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50 + "\n")

    if report and report.sections:
        print(report.sections[0])   