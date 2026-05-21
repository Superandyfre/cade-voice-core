"""Export FSM transition table to Mermaid or DOT format."""

from __future__ import annotations

import sys
from typing import Optional


def export_mermaid(transitions: dict) -> str:
    lines = ["stateDiagram-v2"]
    for event, rule in transitions.items():
        for from_state in sorted(rule.from_states, key=lambda s: s.value):
            label = f"{event}"
            if rule.after:
                label += f" [{', '.join(rule.after)}]"
            lines.append(f"    {from_state.value} --> {rule.to_state.value} : {label}")
    return "\n".join(lines)


def export_mermaid_full(transitions: dict) -> str:
    """Extended Mermaid diagram with FINISH reliability sub-states and journal steps."""
    lines = ["stateDiagram-v2"]
    for event, rule in transitions.items():
        for from_state in sorted(rule.from_states, key=lambda s: s.value):
            label = f"{event}"
            if rule.after:
                label += f" [{', '.join(rule.after)}]"
            lines.append(f"    {from_state.value} --> {rule.to_state.value} : {label}")

    lines.append("")
    lines.append("    state FINISH {")
    lines.append("        [*] --> HookCommit")
    lines.append("        state HookCommit {")
    lines.append("            [*] --> SaveOrderGroup")
    lines.append("            SaveOrderGroup --> AppendEvent : committed")
    lines.append("            AppendEvent --> OutboxPending : committed")
    lines.append("            OutboxPending --> PublishConfirm : committed")
    lines.append("            PublishConfirm --> OutboxPublished : committed")
    lines.append("            OutboxPublished --> TTSFinish : committed")
    lines.append("        }")
    lines.append("        HookCommit --> Pending : journal_committed")
    lines.append("        Pending --> Published : publish_order_confirmed")
    lines.append("        Published --> Delivered : confirmed_ack(delivered)")
    lines.append("        Published --> DeadLetter : confirmed_ack(dead_letter)")
    lines.append("        Published --> Published : outbox_retry")
    lines.append("        Pending --> DeadLetter : max_attempts_exceeded")
    lines.append("    }")
    lines.append("")
    lines.append("    state Recovery {")
    lines.append("        [*] --> ScanSnapshots")
    lines.append("        ScanSnapshots --> SkipCommitted : finish_confirmed/committed")
    lines.append("        ScanSnapshots --> CompletePendingConfirmed : finish_confirmed/pending")
    lines.append("        ScanSnapshots --> CancelIncomplete : other_phase")
    lines.append("        CompletePendingConfirmed --> OutboxPending : ensure_outbox")
    lines.append("    }")
    return "\n".join(lines)


def export_dot(transitions: dict) -> str:
    lines = ["digraph OrderFSM {", '    rankdir=LR;']
    for event, rule in transitions.items():
        for from_state in sorted(rule.from_states, key=lambda s: s.value):
            attrs = []
            if rule.after:
                attrs.append(f'label="{event}\\n[{", ".join(rule.after)}]"')
            else:
                attrs.append(f'label="{event}"')
            lines.append(f'    {from_state.value} -> {rule.to_state.value} [{" ".join(attrs)}];')
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="cade-fsm-graph", description="Export FSM state graph.")
    parser.add_argument("--format", choices=["mermaid", "dot"], default="mermaid")
    parser.add_argument("--full", action="store_true", help="Include reliability sub-states")
    args = parser.parse_args()

    from cade.fsm.order_fsm import OrderSubFSM
    transitions = OrderSubFSM.TRANSITIONS

    if args.format == "dot":
        print(export_dot(transitions))
    elif args.full:
        print(export_mermaid_full(transitions))
    else:
        print(export_mermaid(transitions))


if __name__ == "__main__":
    main()
