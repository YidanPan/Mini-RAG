"""Agent event streaming shared by the terminal CLI and Streamlit UI."""

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

def parse_tool_error(content):
    """Parse the structured error protocol returned by project tools."""
    if not isinstance(content, str) or not content.startswith("ERROR_TYPE:"):
        return None

    error = {}
    for line in content.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)#只切分一次
            error[key.strip()] = value.strip()
    return error


def iter_agent_events(agent, query, config):
    """Yield normalized Agent events without choosing how they are displayed."""
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        stream_mode=["messages", "updates", "custom"],
        version="v2",
    ):
        if chunk["type"] == "updates":
            for update in chunk["data"].values():
                messages = update.get("messages", [])
                if not messages:
                    continue
                message = messages[-1]
                if isinstance(message, AIMessage) and message.tool_calls:
                    for tool_call in message.tool_calls:
                        yield {
                            "type": "tool_call",
                            "name": tool_call["name"],
                            "args": tool_call["args"],
                        }
                elif isinstance(message, ToolMessage):
                    yield {
                        "type": "tool_result",
                        "name": message.name,
                        "content": message.content,
                        "error": parse_tool_error(message.content),
                    }
        elif chunk["type"] == "messages":
            token, _metadata = chunk["data"]
            if not isinstance(token, AIMessageChunk) or token.tool_call_chunks:
                continue
            if token.text:
                yield {"type": "answer_token", "text": token.text}
        elif chunk["type"] == "custom":
            yield {"type": "progress", "message": str(chunk["data"])}


def stream_agent(agent, query, config):
    """Render Agent events in the existing terminal CLI."""
    final_answer_started = False
    for event in iter_agent_events(agent, query, config):
        if event["type"] == "tool_call":
            print(f"\n[Agent] Tool: {event['name']}")
            print("[Arguments]", event["args"])
        elif event["type"] == "tool_result":
            error = event["error"]
            if error:
                print(f"\n[Tool Error: {error.get('ERROR_TYPE', 'UNKNOWN')}]")
                print(f"Tool: {error.get('TOOL', event['name'])}")
                if error.get("SOURCE"):
                    print(f"Source: {error['SOURCE']}")
                print(f"Detail: {error.get('DETAIL', '')}")
                available = error.get("RECOVERABLE", "false").lower() == "true"
                print("Recovery: " + ("available" if available else "unavailable"))
                if error.get("SUGGESTED_ACTION"):
                    print(f"Suggested action: {error['SUGGESTED_ACTION']}")
            else:
                print(f"\n[Tool: {event['name']}]")
                print(event["content"])
        elif event["type"] == "answer_token":
            if not final_answer_started:
                print("\n[Agent Answer]")
                final_answer_started = True #标题输出标志位
            print(event["text"], end="", flush=True)
        elif event["type"] == "progress":
            print(f"\n[Progress] {event['message']}")
    print()
