"""
Skill registry.

The registry is the surface where users (and the closed platform) plug in
specialists. The default registry ships with the canonical six (planner,
memory, solver, critic, verifier, evaluator) plus a synthesizer meta-skill,
matching the activation plans in the planner.

Users can register additional specialists at runtime via `register_skill` or
by instantiating their own `SkillRegistry`.
"""

from __future__ import annotations

from agensflow.schema import SkillSpec
from agensflow.skills import SkillCard


class SkillRegistry:
    """
    In-memory mapping from skill name to SkillSpec, plus a parallel
    mapping from card name to SkillCard.

    Two registries side-by-side because SkillSpec (structural — what
    does this skill consume/produce?) and SkillCard (behavioral — how
    does this skill act?) compose orthogonally:

      - A skill *spec* is the structural contract used by the activation
        planner and router. Every registered skill has one.
      - A skill *card* is the behavioral spec used by the agent factory
        at construction time. A card may be associated with multiple
        spec variants — chunk-9's pattern is one solver card paired
        with three model bindings, where each (card, model) pair has
        its own SkillSpec but shares the card.

    Cards are looked up by their declared name (from the SKILL.md
    frontmatter), not by spec name. Variant binding from spec name to
    card name is handled by `runtime.models.get_card_for_skill`.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}
        self._cards: dict[str, SkillCard] = {}

    def register(self, spec: SkillSpec, *, overwrite: bool = False) -> None:
        if spec.name in self._skills and not overwrite:
            raise ValueError(
                f"Skill {spec.name!r} already registered. "
                f"Pass overwrite=True to replace."
            )
        self._skills[spec.name] = spec

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"No skill registered under {name!r}") from exc

    def has(self, name: str) -> bool:
        return name in self._skills

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def all(self) -> dict[str, SkillSpec]:
        return dict(self._skills)

    # ------------------------------------------------------------------ #
    # SkillCard (behavioral) registration — chunk-9 substrate addition
    # ------------------------------------------------------------------ #

    def register_card(self, card: SkillCard, *, overwrite: bool = False) -> None:
        """Register a SkillCard parsed from a SKILL.md file. Cards are
        keyed by their declared name (from the frontmatter)."""
        if card.name in self._cards and not overwrite:
            raise ValueError(
                f"Skill card {card.name!r} already registered. "
                f"Pass overwrite=True to replace."
            )
        self._cards[card.name] = card

    def register_cards_from_directory(
        self, directory: str, *, overwrite: bool = False
    ) -> int:
        """Bulk-register every SKILL.md card found in `directory`.

        Returns the number of cards registered (useful for diagnostics).
        Honors `overwrite` per-card; collisions raise unless overwrite=True.
        """
        from agensflow.skills import load_skills_from_directory
        cards = load_skills_from_directory(directory)
        for card in cards.values():
            self.register_card(card, overwrite=overwrite)
        return len(cards)

    def get_card(self, name: str) -> SkillCard:
        """Return the SkillCard registered under `name`. Raises KeyError
        if the card hasn't been registered (caller should fall back to
        hardcoded prompts in that case)."""
        try:
            return self._cards[name]
        except KeyError as exc:
            raise KeyError(f"No skill card registered under {name!r}") from exc

    def has_card(self, name: str) -> bool:
        return name in self._cards

    def card_names(self) -> list[str]:
        return list(self._cards.keys())

    def all_cards(self) -> dict[str, SkillCard]:
        return dict(self._cards)


def _build_default_registry() -> SkillRegistry:
    registry = SkillRegistry()
    all_regimes = [
        "straightforward",
        "evidence_heavy",
        "ambiguous",
        "contradictory",
        "high_risk",
        "exploratory",
    ]

    registry.register(
        SkillSpec(
            name="planner",
            kind="agent",
            preconditions=[],
            outputs=["goal", "subproblem", "constraints"],
            handoff_requirements=[],
            preferred_successors=["memory", "solver"],
            confidence_effect=0.05,
            cost_estimate=1.0,
            regime_affinity=all_regimes,  # type: ignore[arg-type]
            branch_compatibility=["memory", "solver"],
            merge_preference="select_best",
        )
    )
    registry.register(
        SkillSpec(
            name="memory",
            kind="agent",
            preconditions=["goal"],
            outputs=["retrieved_context", "evidence"],
            handoff_requirements=["goal"],
            preferred_successors=["solver", "verifier"],
            confidence_effect=0.10,
            cost_estimate=1.2,
            regime_affinity=[
                "evidence_heavy",
                "ambiguous",
                "contradictory",
                "high_risk",
                "exploratory",
            ],
            branch_compatibility=["solver", "verifier", "critic"],
            merge_preference="verifier_gate",
        )
    )
    registry.register(
        SkillSpec(
            name="solver",
            kind="agent",
            preconditions=["subproblem"],
            outputs=["draft_answer"],
            handoff_requirements=["subproblem"],
            preferred_successors=["critic", "verifier", "evaluator"],
            confidence_effect=0.15,
            cost_estimate=1.5,
            regime_affinity=all_regimes,  # type: ignore[arg-type]
            branch_compatibility=["critic", "verifier", "synthesizer"],
            merge_preference="select_best",
        )
    )
    registry.register(
        SkillSpec(
            name="critic",
            kind="agent",
            preconditions=["draft_answer"],
            outputs=["critique"],
            handoff_requirements=["draft_answer"],
            preferred_successors=["solver", "verifier", "evaluator"],
            confidence_effect=-0.05,
            cost_estimate=1.0,
            regime_affinity=["ambiguous", "contradictory", "high_risk"],
            branch_compatibility=["solver", "verifier"],
            merge_preference="critic_select",
        )
    )
    registry.register(
        SkillSpec(
            name="verifier",
            kind="agent",
            preconditions=["draft_answer", "evidence"],
            outputs=["verification"],
            handoff_requirements=["draft_answer", "evidence"],
            preferred_successors=["evaluator"],
            confidence_effect=0.20,
            cost_estimate=1.3,
            regime_affinity=["evidence_heavy", "contradictory", "high_risk"],
            branch_compatibility=["solver", "critic"],
            merge_preference="verifier_gate",
        )
    )
    registry.register(
        SkillSpec(
            name="evaluator",
            kind="meta_skill",
            preconditions=["draft_answer"],
            outputs=["decision"],
            handoff_requirements=["draft_answer"],
            preferred_successors=[],
            confidence_effect=0.0,
            cost_estimate=0.8,
            regime_affinity=all_regimes,  # type: ignore[arg-type]
            branch_compatibility=[],
            merge_preference="select_best",
        )
    )
    registry.register(
        SkillSpec(
            name="synthesizer",
            kind="meta_skill",
            preconditions=["draft_answer"],
            outputs=["merged_answer"],
            handoff_requirements=["draft_answer"],
            preferred_successors=["verifier", "evaluator"],
            confidence_effect=0.10,
            cost_estimate=1.1,
            regime_affinity=["ambiguous", "exploratory"],
            branch_compatibility=["solver", "critic", "verifier"],
            merge_preference="weighted_merge",
        )
    )

    # ----------------------------------------------------------------- #
    # Skill variants — same role, different model bindings.
    #
    # Variants exist so the policy graph can learn per-signature which
    # model fits which query class. Each variant inherits its base skill's
    # preconditions, outputs, and handoff requirements; only the model
    # (declared in runtime.models.SKILL_VARIANT_BINDINGS) differs.
    #
    # The activation planner can include any subset of variants in its
    # selected_skills; the policy graph then accumulates per-(signature,
    # variant) value estimates and converges to the cheapest sufficient
    # variant for each signature class.
    # ----------------------------------------------------------------- #

    # Solver variants — same role, different (family, tier) cells.
    # cost_estimate is a relative hint; actual per-call cost depends on
    # the underlying provider's pricing and the prompt length.
    #
    # Chunk-9 adds the (card × model) cross-product to this list. Each
    # of the three solver cards (concise / chain_of_thought / evidence_first)
    # paired with three model bindings (haiku / fast / mini) gives 9 new
    # variant SkillSpecs. The card-to-model mapping lives in
    # `runtime.models.SKILL_VARIANT_CARDS`; the variant-to-model mapping
    # in `runtime.models.SKILL_VARIANT_BINDINGS`.
    for variant_name, cost in [
        # Chunk 6+7+8 variants (model bindings of the default solver card).
        ("solver_fast", 1.0),         # openai/gpt-5.4-nano — cheap/fast
        ("solver_mini", 1.3),         # openai/gpt-5.4-mini — middle
        ("solver_haiku", 1.5),        # anthropic/claude-haiku-4.5 — middle/capable
        ("solver_qwen_flash", 1.0),   # qwen/qwen3.6-flash — cheap (Qwen line)
        ("solver_qwen_max", 2.5),     # qwen/qwen3.6-max-preview — frontier (1T MoE)
        # Chunk-9 variants: skill cards × model bindings.
        ("solver_concise_haiku", 1.5),
        ("solver_concise_fast",  1.0),
        ("solver_concise_mini",  1.3),
        ("solver_cot_haiku",     1.5),
        ("solver_cot_fast",      1.0),
        ("solver_cot_mini",      1.3),
        ("solver_evidence_haiku", 1.5),
        ("solver_evidence_fast",  1.0),
        ("solver_evidence_mini",  1.3),
    ]:
        registry.register(
            SkillSpec(
                name=variant_name,
                kind="agent",
                preconditions=["subproblem"],
                outputs=["draft_answer"],
                handoff_requirements=["subproblem"],
                preferred_successors=["critic", "verifier", "evaluator"],
                confidence_effect=0.15,
                cost_estimate=cost,
                regime_affinity=all_regimes,  # type: ignore[arg-type]
                branch_compatibility=["critic", "verifier", "synthesizer"],
                merge_preference="select_best",
            )
        )

    # Web-search tools as first-class skills.
    # Outputs match `memory` (evidence + retrieved_context) so downstream
    # agents see external retrievals identically to corpus retrievals.
    # The policy graph learns per signature whether external search is
    # warranted and which provider to use.
    for name, cost in [
        ("web_search_exa", 0.8),     # semantic, technical-strong, ~$0.005/call
        ("web_search_tavily", 0.4),  # general web, cheaper, ~$0.001/call
    ]:
        registry.register(
            SkillSpec(
                name=name,
                kind="skill",
                preconditions=["subproblem"],
                outputs=["evidence", "retrieved_context"],
                handoff_requirements=["subproblem"],
                preferred_successors=[
                    "solver", "solver_fast", "solver_mini", "solver_haiku",
                    "solver_qwen_flash", "solver_qwen_max",
                    # Chunk-9 (card × model) variants
                    "solver_concise_haiku", "solver_concise_fast", "solver_concise_mini",
                    "solver_cot_haiku", "solver_cot_fast", "solver_cot_mini",
                    "solver_evidence_haiku", "solver_evidence_fast", "solver_evidence_mini",
                ],
                confidence_effect=0.10,
                cost_estimate=cost,
                regime_affinity=[
                    "evidence_heavy", "ambiguous", "exploratory", "high_risk",
                ],
                branch_compatibility=["solver"],
                merge_preference="verifier_gate",
            )
        )

    # Verifier variants — verification quality vs. cost.
    for variant_name, cost in [
        ("verifier_fast", 1.0),    # openai/gpt-5.4-nano
        ("verifier_haiku", 1.4),   # anthropic/claude-haiku-4.5
    ]:
        registry.register(
            SkillSpec(
                name=variant_name,
                kind="agent",
                preconditions=["draft_answer", "evidence"],
                outputs=["verification"],
                handoff_requirements=["draft_answer", "evidence"],
                preferred_successors=["evaluator"],
                confidence_effect=0.20,
                cost_estimate=cost,
                regime_affinity=["evidence_heavy", "contradictory", "high_risk"],
                branch_compatibility=["solver", "critic"],
                merge_preference="verifier_gate",
            )
        )

    return registry


default_registry: SkillRegistry = _build_default_registry()


def register_skill(spec: SkillSpec, *, overwrite: bool = False) -> None:
    """Register a skill in the default registry."""
    default_registry.register(spec, overwrite=overwrite)
