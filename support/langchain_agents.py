from django.conf import settings
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import wrap_tool_call
from .event_queue import publish, DONE
from .langchain_tools import get_order_details, get_refund_history, check_delivery_status, search_knowledge_base, \
    get_customer_risk_profile
from .agents import SUPPORT_SYSTEM_PROMPT, MANAGER_SYSTEM_PROMPT, RISK_SYSTEM_PROMPT
from .models import Conversation, AgentLog
from langchain.tools import tool

# Initialize Anthropic client
llm = ChatAnthropic(model=settings.ANTHROPIC_MODEL, api_key=settings.ANTHROPIC_API_KEY,thinking={"type": "disabled"})

def run_support_agent_langchain(user_message, conversation_id, order_id, user_id):
    conv = Conversation.objects.get(id=conversation_id)
    @tool
    def escalate_to_manager(case_summary: str) -> dict:
        """Escalate the case to manager for refund decision. Always include customer's user_id in the case summary so manager can assess fraud risk accurately."""
        return run_manager_agent_langchain(case_summary, conversation_id)
    config = {"configurable": {"thread_id": str(conversation_id)}}
    contextual_message = f"[Context: This conversation is about Order #{order_id}, user: {user_id}]{user_message}"

    @wrap_tool_call
    def log_tool_calls_middleware(request, handler):
        # Before tool execution
        print("About to call a tool")
        print("request==>", request)
        # log tool call
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]
        event = {"type": "tool_call", "message": f"Calling tool {tool_name} with {tool_args}"}
        publish(conversation_id, event)
        AgentLog.objects.create(
            conversation=conv,
            event_type="tool_call",
            message=f"Calling tool {tool_name} with {tool_args}"
        )

        result = handler(request)
        # After tool execution
        event = {"type": "tool_result", "message": f"{tool_name} returned: {str(result.content)[:200]}"}
        publish(conversation_id, event)
        # log tool result
        AgentLog.objects.create(
            conversation=conv,
            event_type="tool_result",
            message=f"{tool_name} returned: {str(result.content)[:200]}"
        )
        return result

    SUPPORT_TOOLS = [get_order_details, get_refund_history, check_delivery_status, search_knowledge_base]
    checkpointer = InMemorySaver()
    support_agent = create_agent(
        model=llm,
        tools=SUPPORT_TOOLS + [escalate_to_manager],
        system_prompt=SUPPORT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[log_tool_calls_middleware]
    )

    result = support_agent.invoke(
        {"messages": [{"role": "user", "content": contextual_message}]},
        config=config,
    )
    final_reply = result["messages"][-1].content
    event = {"type": "final", "message": final_reply}
    publish(conversation_id, event)
    AgentLog.objects.create(
        conversation=conv,
        event_type="final",
        message=final_reply)
    publish(conversation_id, DONE)
    return final_reply


def run_manager_agent_langchain(case_summary, conversation_id):
    conv = Conversation.objects.get(id=conversation_id)
    @tool
    def assess_fraud_risk(user_id: int) -> dict:
        """Consult the risk agent to assess fraud risk for a customer. Use this when refund request looks suspicious or customer has multiple refund requests. Pass the user_id to get a risk verdict."""
        return run_risk_agent_langchain(user_id, conversation_id)
    event = {"type": "manager", "message": f"Case received for review: {case_summary[:200]}"}
    publish(conversation_id, event=event)
    AgentLog.objects.create(conversation=conv, event_type="manager",
                            message=f"Case received for review: {case_summary[:200]}")
    # middleware
    @wrap_tool_call
    def log_manager_tool_calls_middleware(request, handler):
        # before executing the tool
        event = {"type": "manager", "message": "Consulting risk agent for fraud assessment"}
        publish(conversation_id, event=event)
        # log consulting risk agent
        AgentLog.objects.create(
            conversation=conv,
            event_type="manager",
            message="Consulting risk agent for fraud assessment"
        )
        result = handler(request)
        #after executing the tool
        return result

    manager_agent = create_agent(
        model=llm,
        system_prompt=MANAGER_SYSTEM_PROMPT,
        tools=[assess_fraud_risk],
        middleware=[log_manager_tool_calls_middleware]
    )
    result = manager_agent.invoke({
        "messages": [{"role": "user", "content": case_summary}]
    })
    decision = result["messages"][-1].content
    event = {"type": "manager", "message": f"Decision: {decision[:200]}"}
    publish(conversation_id, event=event)
    AgentLog.objects.create(conversation=conv, event_type="manager", message=f"Decision: {decision[:200]}")
    return decision


def run_risk_agent_langchain(user_id, conversation_id):
    conv = Conversation.objects.get(id=conversation_id)
    event = {"type": "risk", "message": f"Starting fraud assessment for user: {user_id}"}
    publish(conversation_id, event=event)
    # log assessment started
    AgentLog.objects.create(conversation=conv, event_type="risk",
                            message=f"Starting fraud assessment for user: {user_id}")
    @wrap_tool_call
    def log_risk_tool_calls_middleware(request, handler):
        tool_name = request.tool_call["name"]
        event = {"type": "risk", "message": f"Calling {tool_name} to get customer risk profile...."}
        publish(conversation_id, event=event)
        AgentLog.objects.create(conversation=conv, event_type="risk",
                                message=f"Calling {tool_name} to get customer risk profile....")
        result =handler(request)
        return result
    risk_agent = create_agent(
        model=llm,
        system_prompt=RISK_SYSTEM_PROMPT,
        tools=[get_customer_risk_profile],
        middleware=[log_risk_tool_calls_middleware]
    )
    result = risk_agent.invoke({
        "messages": [{"role": "user", "content": f"Please assess the fraud risk for user ID {user_id}. Use your tool to get their profile and return a verdict."}]

    })
    verdict = result["messages"][-1].content
    event = {"type": "risk", "message": f"Verdict: {verdict[:200]}"}
    publish(conversation_id, event=event)
    AgentLog.objects.create(conversation=conv, event_type="risk", message=f"Verdict: {verdict[:200]}")
    return verdict