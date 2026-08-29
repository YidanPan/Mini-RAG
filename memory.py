from langchain_core.messages import HumanMessage, AIMessage

#实现对话历史的保存和操作
class ConversationMemory:

    def __init__(self):
        self.messages = []

    def get_history(self):
        return self.messages

    def add_user_message(self, content):
        self.messages.append(
            HumanMessage(content=content)
        )

    def add_ai_message(self, content):
        self.messages.append(
            AIMessage(content=content)
        )

    def clear(self):
        self.messages = []