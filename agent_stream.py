from langchain_core.messages import (AIMessage,AIMessageChunk,ToolMessage)

#专门负责 Agent Streaming 的终端渲染，把 agent 执行过程中的工具调用信息和最终回答的 token 实时打印到终端
def stream_agent(agent,query,config):
    final_answer_started = False #标志位，判断是否已经打印过标题

    for chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        config=config,
        stream_mode=["messages","updates","custom"],#同时启用两种stream，messages是返回token stream，update是每个节点执行完后对state的更新
        version="v2"
    ):
        if chunk["type"] == "updates":
            for node_name, update in (
                chunk["data"].items()
            ):
                if "messages" not in update:
                    continue
                message = (update["messages"][-1])#取更新后的最后一条消息

                if isinstance(message,AIMessage) and message.tool_calls:
                    for tool_call in (message.tool_calls):
                        #打印工具名和参数
                        print(
                            f"\n[Agent] Tool: "
                            f"{tool_call['name']}"
                        )
                        print(
                            "[Arguments]",
                            tool_call["args"]
                        )
                elif isinstance(message,ToolMessage):
                    print(
                        f"\n[Tool: "
                        f"{message.name}]"
                    )
                    print(message.content)

        elif chunk["type"] == "messages":
            token, metadata = chunk["data"]

            if not isinstance(token,AIMessageChunk):
                continue
            if token.tool_call_chunks:
                continue

            text = token.text
            if not text:
                continue
            if not final_answer_started:
                print("\n[Agent Answer]")
                final_answer_started = True
            print(
                text,
                end="",
                flush=True#逐字打印
            )
        elif chunk["type"]=="custom":
            print(
                f"\n[Progress] "
                f"{chunk['data']}"
            )
    print()