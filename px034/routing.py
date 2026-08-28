from __future__ import annotations

from .schema import Route, SupportState


DOMAIN_STRATEGY_BY_TASK = {
    "entity_linking": "extract_then_retrieve",
    "entity_attribution": "decompose_then_retrieve",
    "multi_document_synthesis": "cskg_guided",
    "multi_doc_synthesis": "cskg_guided",
}


def route_for_state(state: str, task: str) -> dict[str, str]:
    """Frozen default policy; alternatives belong in preregistered config."""
    normalized = SupportState(state)
    if normalized in {SupportState.DECISIVE, SupportState.COMPLEMENTARY}:
        strategy = DOMAIN_STRATEGY_BY_TASK.get(task)
        if not strategy:
            raise ValueError(f"unknown CTIConnect task category: {task}")
        return {"route": Route.DOMAIN_SPECIFIC.value, "strategy": strategy}
    if normalized in {SupportState.CONFLICTING, SupportState.STALE, SupportState.OTHER}:
        return {"route": Route.ABSTAIN.value, "strategy": "review_or_refresh"}
    return {"route": Route.CLOSED_BOOK.value, "strategy": "no_retrieval"}


def selected_correct(route: str, outcomes: dict[str, bool]) -> bool | None:
    if route == Route.ABSTAIN.value:
        return None
    key = {
        Route.CLOSED_BOOK.value: "closed_book_correct",
        Route.VANILLA_RAG.value: "vanilla_rag_correct",
        Route.DOMAIN_SPECIFIC.value: "domain_specific_correct",
    }[Route(route)]
    return bool(outcomes[key])
