from models.state import ResearchState, Evaluation
from models.enums import DecisionEnum

def evaluator_node(state: ResearchState) -> ResearchState:
    print("Evaluator Node")

    # Dummy evaluation
    state.evaluation = Evaluation(
        confidence_score=0.7,
        decision=DecisionEnum.proceed
    )

    return state