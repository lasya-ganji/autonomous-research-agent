from observability.langsmith_config import setup_langsmith
from app.agent_runner import run_agent

setup_langsmith()

if __name__ == "__main__":
    result = run_agent("What are the latest advancements in renewable energy technologies?")
    print("\nFull Result:\n", result)
    report = result.get("report")
    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50 + "\n")

    if report and report.sections:
        print(report.sections[0])   