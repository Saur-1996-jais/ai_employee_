from anthropic import Anthropic
from django.conf import settings
from .tools import get_order_details, get_refund_history, check_delivery_status
from .models import Conversation, Message, AgentLog
# Initialize Anthropic client
client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

anthropic_model = settings.ANTHROPIC_MODEL

# Support System Prompt ----> Maya's job description
# Its actually prompt that tells how you talk to claude or chatgpt but with the proper structure
SUPPORT_SYSTEM_PROMPT = """
You are Maya, a customer support agent at CoolBreeze AC.
You help customers with issues related to their orders.

Your responsibilities:
- Always use your tools to gather facts at before responding
- Check order details when customer mentions their order
- Check refund history before making any refund decisions
- Be empathetic but honest

Your Personality:
- Friendly and Professional
- Patient even when customer is angry
- Clear and concise in your replies

Important rules:
- Always check order details first before responding
- Never approve or deny a refund yourself
- If refund decision is needed - tell customer you are checking with your team 
"""

# Support Tools ---> Tool schemas, that ai agents will read
SUPPORT_TOOLS = [
    {
        "name": "get_order_details",
        "description": "Fetch complete order details including status, carrier, tracking number and days since order was placed. Use this when customer mention their order or complains about delivery.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "integer",
                    "description": "The Order ID to look up",
                }
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "get_refund_history",
        "description": "Get complete refund history for a user.Use this before making any refund related decisions",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "The User ID to check refund history for"
                }
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "check_delivery_status",
        "description": "Check current delivery status using tracking number and carrier. Use this when customer complains about delayed or missing delivery",
        "input_schema": {
            "type": "object",
            "properties": {
                "tracking_number": {
                    "type": "string",
                    "description": "The shipment tracking number"
                },
                "carrier": {
                    "type": "string",
                    "description": "The carrier name for example BlueDart or Delhivery"
                }
            },
            "required": ["tracking_number", "carrier"]
        }
    }
]


# Execute_tool() ---> bridge between python function and claude
def execute_tool(tool_name, tool_input):
    if tool_name == "get_order_details":
        return get_order_details(tool_input["order_id"])
    if tool_name == "get_refund_history":
        return get_refund_history(tool_input["user_id"])
    if tool_name == "check_delivery_status":
        return check_delivery_status(tool_input["tracking_number"], tool_input["carrier"])

# Agent Loop ---> while loop that loops until the task is done
def run_support_agent(user_message, conversation_id, order_id, user_id):
    conv = Conversation.objects.get(id=conversation_id)
    conversation_messages = []
    for msg in conv.messages.order_by("created_at"):
        conversation_messages.append({
            "role": msg.role,
            "content": msg.content,
        })

    while True:
        # send this conversation to LLM
        response = client.messages.create(
            model=anthropic_model,
            max_tokens=1024,
            system=SUPPORT_SYSTEM_PROMPT + f"\n\nContext: This conversation is about Order #{order_id}, user: {user_id}",
            messages=conversation_messages,
            tools=SUPPORT_TOOLS,
            thinking={"type": "disabled"},
        )

        print('stop_reason ==>', response.stop_reason)
        print("Content==>", response.content)

        if response.stop_reason == ("tool_use"):
            tool_result = []
            for block in response.content:
                if block.type == "tool_use":
                    print("tool call ==>", block.name)
                    print("tool input ==>", block.input)

                    #execute the tools
                    result = execute_tool(block.name, block.input)
                    print("Tool result", result)

                    tool_result.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
            conversation_messages.append({
                "role": "assistant",
                "content": response.content
            })
            conversation_messages.append({
                "role": "user",
                "content": tool_result
            })
        else:
            # Extract all text blocks safely
            final_text = "\n".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            print("Final response ==>", final_text)

            return response.content[0].text