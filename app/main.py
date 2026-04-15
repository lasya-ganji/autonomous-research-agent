from observability.langsmith_config import setup_langsmith
from app.agent_runner import run_agent

setup_langsmith()

if __name__ == "__main__":
    query = input("Enter your research query: ")

    result = run_agent(query)

    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50 + "\n")

    report = result.get("report")

    if report and report.sections:
        print(report.sections[0])
    else:
        print("No report generated.")