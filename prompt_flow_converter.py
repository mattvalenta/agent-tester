"""
Prompt Flow Converter — Single-Prompt LLM Orchestration

This module converts Pipecat flow JSON configurations into single, structured
natural-language system prompts. The LLM tracks conversation state using
instructions embedded in the system message, with function-calling tools
registered directly on the LLM.

The converted prompt covers:
    - Flow metadata (name, version)
    - Agent role & personality
    - Full conversation flow with all node fields:
      id, label, type, description, functions (with full schemas,
      decision routing, and parameters), pre/post actions
    - Behavioral guidelines for voice-optimized output

Supports two flow formats:
    1. Array-based: nodes is a list of objects with id/type/label/description/
       functions (rich objects with decision routing)
    2. Dict-based: nodes is a dict keyed by node name with messages/functions
       (string lists)/pre_actions/post_actions, plus top-level initial_node
"""

import hashlib
import json
from loguru import logger

_generic_flow_prompt_cache: dict[str, str] = {}

def clear_flow_prompt_cache():
    _generic_flow_prompt_cache.clear()
    logger.info("🗑️ Generic flow prompt cache cleared")


def _is_array_format(flow_config: dict) -> bool:
    nodes = flow_config.get("nodes", [])
    return isinstance(nodes, list)


def _render_function_simple(fn_name: str) -> str:
    return f"  - {fn_name}"


def _is_routing_function(fn: dict) -> bool:
    return "decision" in fn or "next_node_id" in fn


def _render_function_rich(fn: dict) -> str:
    if _is_routing_function(fn):
        return _render_routing_instruction(fn)
    return _render_tool_schema(fn)


def _render_tool_schema(fn: dict) -> str:
    lines: list[str] = []
    name = fn.get("name", "unnamed")
    desc = fn.get("description", "")
    lines.append(f"  #### Function: {name}")
    if desc:
        lines.append(f"  {desc}")

    required = fn.get("required", [])
    if required:
        lines.append(f"  - Required parameters: {', '.join(required)}")

    _render_properties_block(fn.get("properties", {}), lines)

    return "\n".join(lines)


def _render_routing_instruction(fn: dict) -> str:
    lines: list[str] = []
    name = fn.get("name", "unnamed")
    desc = fn.get("description", "")
    label = name.replace("_", " ").replace("-", " ").title()

    next_node = fn.get("next_node_id")
    decision = fn.get("decision")

    if decision:
        lines.append(f"  **Routing step — {label} ({name}):**")
        if desc:
            lines.append(f"  {desc}")

        required = fn.get("required", [])
        if required:
            lines.append(f"  Key factors to assess: {', '.join(required)}")

        _render_properties_block(fn.get("properties", {}), lines)

        action = decision.get("action", "")
        if action:
            lines.append(f"  Decision action: {action}")

        conditions = decision.get("conditions", [])
        default_target = decision.get("default_next_node_id", "")
        if conditions or default_target:
            lines.append("  Based on the conversation, route as follows:")
            for cond in conditions:
                val = cond.get("value", "")
                op = cond.get("operator", "==")
                target = cond.get("next_node_id", "")
                target_label = target.replace("_", " ").replace("-", " ").title()
                lines.append(f"    - If {op} \"{val}\" → proceed to **{target_label}** ({target})")
            if default_target:
                default_label = default_target.replace("_", " ").replace("-", " ").title()
                lines.append(f"    - Otherwise → proceed to **{default_label}** ({default_target})")
    elif next_node:
        target_label = next_node.replace("_", " ").replace("-", " ").title()
        lines.append(f"  **When {label} ({name}):** {desc if desc else 'proceed to next stage'}")

        required = fn.get("required", [])
        if required:
            lines.append(f"  Key factors to assess: {', '.join(required)}")

        _render_properties_block(fn.get("properties", {}), lines)

        lines.append(f"    → Proceed to **{target_label}** ({next_node})")

    return "\n".join(lines)


def _render_properties_block(properties: dict, lines: list[str]) -> None:
    if not properties:
        return
    lines.append("  - Parameters:")
    known_keys = {"description", "enum", "pattern", "type", "format", "default", "items"}
    for prop_name, prop_def in properties.items():
        prop_desc = prop_def.get("description", "")
        prop_type = prop_def.get("type", "")
        enum_vals = prop_def.get("enum", [])
        pattern = prop_def.get("pattern", "")
        fmt = prop_def.get("format", "")
        default = prop_def.get("default")
        items = prop_def.get("items")
        # Unescape any JSON-escaped bold markers from property names (e.g. \"\\*\\*name\\*\\*\" → \"name\")
        clean_name = prop_name.replace("\\\\*\\\\*", "*").replace("\\*\\*", "*")
        parts = [f"    - {clean_name}"]
        if prop_type:
            parts.append(f" (type: {prop_type})")
        if enum_vals:
            parts.append(f" (enum: {', '.join(str(v) for v in enum_vals)})")
        if pattern:
            parts.append(f" (pattern: {pattern})")
        if fmt:
            parts.append(f" (format: {fmt})")
        if default is not None:
            parts.append(f" (default: {default})")
        if items:
            parts.append(f" (items: {json.dumps(items)})")
        extra_keys = set(prop_def.keys()) - known_keys
        for ek in sorted(extra_keys):
            parts.append(f" ({ek}: {json.dumps(prop_def[ek])})")
        if prop_desc:
            parts.append(f" — {prop_desc}")
        lines.append("".join(parts))


def _convert_array_format(flow_config: dict) -> str:
    nodes = flow_config.get("nodes", [])
    edges = flow_config.get("edges", [])
    meta = flow_config.get("meta", {})

    node_by_id: dict[str, dict] = {}
    for n in nodes:
        node_by_id[n["id"]] = n

    edge_map: dict[str, list[tuple[str, str]]] = {}
    for e in edges:
        src = e.get("source", "")
        edge_map.setdefault(src, []).append((e.get("label", ""), e.get("target", "")))

    initial_node_id = None
    for n in nodes:
        if n.get("type") == "initial":
            initial_node_id = n["id"]
            break
        if n.get("data", {}).get("type") == "initial":
            initial_node_id = n["id"]
            break
    if not initial_node_id and nodes:
        initial_node_id = nodes[0]["id"]

    visited: set[str] = set()
    ordered_ids: list[str] = []

    def _walk(nid: str) -> None:
        if nid in visited or nid not in node_by_id:
            return
        visited.add(nid)
        ordered_ids.append(nid)
        for _, target in edge_map.get(nid, []):
            _walk(target)
        node = node_by_id[nid]
        data = node.get("data", node)
        all_fns = node.get("functions", []) or data.get("functions", [])
        for fn in all_fns:
            if isinstance(fn, dict):
                next_id = fn.get("next_node_id")
                if next_id:
                    _walk(next_id)
                decision = fn.get("decision", {})
                if decision:
                    for cond in decision.get("conditions", []):
                        _walk(cond.get("next_node_id", ""))
                    default = decision.get("default_next_node_id")
                    if default:
                        _walk(default)

    if initial_node_id:
        _walk(initial_node_id)
    for n in nodes:
        if n["id"] not in visited:
            _walk(n["id"])

    role_text = ""
    if initial_node_id:
        init_node = node_by_id.get(initial_node_id, {})
        init_data = init_node.get("data", init_node)
        for rm in init_data.get("role_messages", []):
            if rm.get("role") == "system":
                role_text = rm.get("content", "")
                break
        if not role_text:
            for rm in init_data.get("role_messages", []):
                role_text = rm.get("content", "")
                break

    sections: list[str] = [
        "# SYSTEM PROMPT — SINGLE-PROMPT VOICE AGENT",
        "",
    ]

    if meta:
        meta_parts = []
        for mk, mv in meta.items():
            meta_parts.append(f"{mk}: {mv}")
        if meta_parts:
            sections.append(f"**{' | '.join(meta_parts)}**")
            sections.append("")

    if role_text:
        sections.append("## ROLE")
        sections.append(role_text)
        sections.append("")

    sections.append("## CONVERSATION FLOW")
    sections.append("")

    stage_num = 0
    for nid in ordered_ids:
        stage_num += 1
        node = node_by_id[nid]
        data = node.get("data", node)

        node_label = node.get("label", "") or data.get("label", "")
        node_type = node.get("type", "") or data.get("type", "")

        type_tag = ""
        if node_type == "initial":
            type_tag = " [START]"
        elif node_type == "end":
            type_tag = " [END]"

        header = f"### Stage {stage_num} — {node_label or nid} (id: {nid}, type: {node_type or 'node'}){type_tag}"
        sections.append(header)

        description = node.get("description", "") or data.get("description", "")
        if description:
            sections.append(f"**Description:** {description}")

        for rm in data.get("role_messages", []):
            content = rm.get("content", "").strip()
            if content and content != role_text:
                sections.append(content)

        for msg in data.get("messages", []):
            if isinstance(msg, dict):
                content = msg.get("content", "").strip()
            else:
                content = str(msg).strip()
            if content:
                sections.append(content)

        for msg in data.get("task_messages", []):
            if isinstance(msg, dict):
                content = msg.get("content", "").strip()
            else:
                content = str(msg).strip()
            if content:
                sections.append(content)

        functions = node.get("functions", []) or data.get("functions", [])
        if functions:
            routing_fns = [fn for fn in functions if isinstance(fn, dict) and _is_routing_function(fn)]
            tool_fns = [fn for fn in functions if isinstance(fn, dict) and not _is_routing_function(fn)]
            simple_fns = [fn for fn in functions if not isinstance(fn, dict)]

            if routing_fns:
                sections.append("- **Conversation routing:**")
                for fn in routing_fns:
                    sections.append(_render_routing_instruction(fn))
            if tool_fns or simple_fns:
                sections.append("- **Functions:**")
                for fn in tool_fns:
                    sections.append(_render_tool_schema(fn))
                for fn in simple_fns:
                    sections.append(_render_function_simple(fn))

        pre_actions = node.get("pre_actions", []) or data.get("pre_actions", [])
        if pre_actions:
            sections.append(f"- Pre-actions: {', '.join(str(a) for a in pre_actions)}")

        post_actions = node.get("post_actions", []) or data.get("post_actions", [])
        if post_actions:
            sections.append(f"- Post-actions: {', '.join(str(a) for a in post_actions)}")

        transitions = edge_map.get(nid, [])
        if transitions:
            sections.append("- Edge transitions:")
            for label, target in transitions:
                target_label = target.replace("_", " ").replace("-", " ").title()
                if label:
                    sections.append(f"  - [{label}] → {target_label}")
                else:
                    sections.append(f"  - → {target_label}")

        sections.append("")

    sections.append("## BEHAVIORAL GUIDELINES")
    sections.append("")
    sections.append("- Keep responses SHORT: 1-2 sentences maximum per turn.")
    sections.append("- ALWAYS end with a question to yield the turn back to the customer.")
    sections.append("- Use turn-yielding phrases: \"Sound good?\", \"Does that make sense?\", \"Got it?\"")
    sections.append("- Break information into digestible chunks — never monologue.")
    sections.append("- Your output will be converted to audio so avoid special characters.")

    return "\n".join(sections)


def _convert_dict_format(flow_config: dict) -> str:
    nodes = flow_config.get("nodes", {})
    initial_node = flow_config.get("initial_node", "")
    meta = flow_config.get("meta", {})

    if initial_node and initial_node in nodes:
        ordered_keys = [initial_node] + [k for k in nodes if k != initial_node]
    else:
        ordered_keys = list(nodes.keys())

    role_text = ""
    if initial_node and initial_node in nodes:
        init_data = nodes[initial_node]
        for rm in init_data.get("role_messages", []):
            if isinstance(rm, dict):
                if rm.get("role") == "system":
                    role_text = rm.get("content", "")
                    break

    sections: list[str] = [
        "# SYSTEM PROMPT — SINGLE-PROMPT VOICE AGENT",
        "",
    ]

    if meta:
        meta_parts = []
        for mk, mv in meta.items():
            meta_parts.append(f"{mk}: {mv}")
        if meta_parts:
            sections.append(f"**{' | '.join(meta_parts)}**")
            sections.append("")

    if initial_node:
        sections.append(f"**Initial node:** {initial_node}")
        sections.append("")

    if role_text:
        sections.append("## ROLE")
        sections.append(role_text)
        sections.append("")

    sections.append("## CONVERSATION FLOW")
    sections.append("")

    has_any_dict_fn = any(
        isinstance(fn, dict)
        for nd in nodes.values()
        for fn in nd.get("functions", [])
    )

    terminal_nodes: set[str] = set()
    for nk, nd in nodes.items():
        node_type = nd.get("type", "")
        if node_type == "end":
            terminal_nodes.add(nk)
            continue
        if not has_any_dict_fn:
            continue
        has_outgoing = False
        for fn in nd.get("functions", []):
            if isinstance(fn, dict):
                if fn.get("next_node_id") or fn.get("decision"):
                    has_outgoing = True
                    break
        if not has_outgoing and nk != initial_node:
            terminal_nodes.add(nk)

    stage_num = 0
    for node_key in ordered_keys:
        stage_num += 1
        data = nodes[node_key]

        node_label = data.get("label", "") or node_key.replace("_", " ").replace("-", " ").title()
        node_type = data.get("type", "node")
        is_initial = (node_key == initial_node)
        is_terminal = node_key in terminal_nodes

        type_tag = ""
        if is_initial:
            type_tag = " [START]"
        elif is_terminal or node_type == "end":
            type_tag = " [END]"

        header = f"### Stage {stage_num} — {node_label} (id: {node_key}, type: {node_type}){type_tag}"
        sections.append(header)

        description = data.get("description", "")
        if description:
            sections.append(f"**Description:** {description}")

        for rm in data.get("role_messages", []):
            if isinstance(rm, dict):
                content = rm.get("content", "").strip()
                if content and content != role_text:
                    sections.append(content)

        for msg in data.get("messages", []):
            if isinstance(msg, dict):
                content = msg.get("content", "").strip()
            else:
                content = str(msg).strip()
            if content:
                sections.append(content)

        for msg in data.get("task_messages", []):
            if isinstance(msg, dict):
                content = msg.get("content", "").strip()
            else:
                content = str(msg).strip()
            if content:
                sections.append(content)

        functions = data.get("functions", [])
        if functions:
            routing_fns = [fn for fn in functions if isinstance(fn, dict) and _is_routing_function(fn)]
            tool_fns = [fn for fn in functions if isinstance(fn, dict) and not _is_routing_function(fn)]
            simple_fns = [fn for fn in functions if not isinstance(fn, dict)]

            if routing_fns:
                sections.append("- **Conversation routing:**")
                for fn in routing_fns:
                    sections.append(_render_routing_instruction(fn))
            if tool_fns or simple_fns:
                sections.append("- **Functions:**")
                for fn in tool_fns:
                    sections.append(_render_tool_schema(fn))
                for fn in simple_fns:
                    sections.append(_render_function_simple(fn))

        pre_actions = data.get("pre_actions", [])
        if pre_actions:
            sections.append(f"- Pre-actions: {', '.join(str(a) for a in pre_actions)}")

        post_actions = data.get("post_actions", [])
        if post_actions:
            sections.append(f"- Post-actions: {', '.join(str(a) for a in post_actions)}")

        sections.append("")

    sections.append("## BEHAVIORAL GUIDELINES")
    sections.append("")
    sections.append("- Keep responses SHORT: 1-2 sentences maximum per turn.")
    sections.append("- ALWAYS end with a question to yield the turn back to the customer.")
    sections.append("- Use turn-yielding phrases: \"Sound good?\", \"Does that make sense?\", \"Got it?\"")
    sections.append("- Break information into digestible chunks — never monologue.")
    sections.append("- Your output will be converted to audio so avoid special characters.")

    return "\n".join(sections)


def convert_generic_flow_to_prompt(flow_config: dict) -> str:
    """Convert any Pipecat flow configuration into a structured system prompt
    by dynamically reading the flow's actual nodes, edges, and all associated
    metadata including functions, decision routing, descriptions, pre/post
    actions, labels, and types.

    Works with both array-based and dict-based flow layouts. Results are cached
    by a hash of the flow config for process-lifetime reuse.
    """
    cache_key = hashlib.md5(json.dumps(flow_config, sort_keys=True).encode()).hexdigest()
    if cache_key in _generic_flow_prompt_cache:
        cached = _generic_flow_prompt_cache[cache_key]
        logger.info(f"⚡ Using cached generic flow prompt ({len(cached)} chars)")
        return cached

    logger.info("Converting generic flow configuration to single-prompt instructions")

    if _is_array_format(flow_config):
        prompt = _convert_array_format(flow_config)
    else:
        prompt = _convert_dict_format(flow_config)

    _generic_flow_prompt_cache[cache_key] = prompt
    logger.info(f"Generated and cached generic prompt: {len(prompt)} characters, ~{len(prompt.split())} words")
    return prompt
