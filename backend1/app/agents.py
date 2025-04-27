import json
import re
import logging
from typing import Dict, Any, Tuple
from .config import config
from .rag_pipeline import RAGPipeline
from autogen import AssistantAgent, UserProxyAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define tool schemas with detailed instructions
TOOLS = [
    {
        "name": "summarize",
        "description": "Generate a comprehensive summary of the provided context, covering research objectives, methodology, key findings, conclusions, and limitations.",
        "parameters": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "The text context to summarize."
                }
            },
            "required": ["context"]
        },
        "instructions": """Provide a structured summary with the following sections:
1. **Research Objectives**: State the main goals and questions addressed.
2. **Methodology**: Describe the approach and techniques used.
3. **Key Findings**: Highlight the most significant results and insights.
4. **Conclusions**: Summarize the implications and contributions.
5. **Limitations**: Note any mentioned constraints or future work.
Use clear headings and concise paragraphs. Do not include page references or citations.
Example:
**Research Objectives**: The study aimed to improve machine translation using a novel architecture.
**Methodology**: A transformer model with self-attention layers was implemented and trained on multilingual datasets.
**Key Findings**: The model achieved a 15% improvement in BLEU scores compared to baselines.
**Conclusions**: The transformer offers significant advancements in translation quality.
**Limitations**: The approach is computationally intensive, limiting its use on low-resource devices."""
    },
    {
        "name": "abstract",
        "description": "Compose a 200-250 word academic abstract covering problem statement, methodology, results, and significance.",
        "parameters": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "The text context to generate an abstract from."
                }
            },
            "required": ["context"]
        },
        "instructions": """Write a 200-250 word academic abstract that includes:
1. **Problem Statement**: The research gap or issue addressed.
2. **Methodology**: The approach used to investigate.
3. **Results**: The key findings and outcomes.
4. **Significance**: The implications and contributions.
Use a formal academic tone and ensure the abstract is a standalone summary. Do not include citations or page references.
Example:
This study addresses the challenge of efficient sequence transduction in neural machine translation, where traditional models struggle with long-range dependencies. We propose a transformer architecture that leverages multi-head self-attention to capture contextual relationships across input sequences. The methodology involves training a six-layer encoder-decoder model on a multilingual corpus, optimizing for BLEU scores. Results demonstrate a 15% improvement in translation accuracy over recurrent neural networks, with reduced training time. The transformer’s ability to parallelize computations enhances scalability, making it suitable for large-scale applications. This work contributes to advancing machine translation systems, offering a robust framework for future research in natural language processing. Limitations include high computational requirements, suggesting avenues for optimizing resource efficiency."""
    },
    {
        "name": "key_findings",
        "description": "Extract and list significant quantitative results, qualitative insights, novel contributions, and surprising results from the context.",
        "parameters": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "The text context to extract key findings from."
                }
            },
            "required": ["context"]
        },
        "instructions": """List the key findings as concise, standalone statements, categorized as:
1. **Quantitative Results**: Numerical results with metrics (e.g., accuracy, scores).
2. **Qualitative Insights**: Important observations or trends.
3. **Novel Contributions**: New methods, models, or ideas introduced.
4. **Surprising Results**: Unexpected outcomes.
Use bullet points under each category. Do not include page references or general descriptions unless they are specific findings.
Example:
- **Quantitative Results**:
  - Achieved a 15% improvement in BLEU scores on translation tasks.
  - Reduced training time by 30% compared to RNNs.
- **Qualitative Insights**:
  - Self-attention mechanisms better capture long-range dependencies.
- **Novel Contributions**:
  - Introduced a transformer model with multi-head attention.
- **Surprising Results**:
  - Model performed well on low-resource languages unexpectedly."""
    },
    {
        "name": "challenges",
        "description": "Identify and explain methodological limitations, data constraints, theoretical boundaries, practical challenges, and future work from the context.",
        "parameters": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "The text context to analyze for challenges and limitations."
                }
            },
            "required": ["context"]
        },
        "instructions": """List and explain the challenges and limitations, categorized as:
1. **Methodological Issues**: Limitations in the research approach.
2. **Data Constraints**: Issues with data collection or quality.
3. **Theoretical Boundaries**: Limitations of the theoretical framework.
4. **Practical Challenges**: Difficulties in implementation.
5. **Future Work**: Suggested directions for future research.
For each, provide a brief description, its impact, and any author-proposed solutions or your suggestions. Use bullet points under each category. Do not include page references.
Example:
- **Methodological Issues**:
  - Reliance on self-attention increases computational complexity; impacts scalability; authors suggest pruning techniques.
- **Data Constraints**:
  - Limited multilingual data for low-resource languages; affects generalizability; suggest data augmentation.
- **Theoretical Boundaries**:
  - Model assumes fixed-length inputs; limits handling of streaming data; propose dynamic input handling.
- **Practical Challenges**:
  - High memory usage in training; hinders deployment on edge devices; suggest model compression.
- **Future Work**:
  - Explore lightweight transformer variants for resource-constrained environments."""
    }
]

async def tool_call(tool_name: str, context: str) -> str:
    """Execute a tool call using AutoGen agents with the Gemma3-27B model."""
    try:
        logger.info(f"Executing tool: {tool_name}")

        # Find the tool schema
        tool_schema = next((tool for tool in TOOLS if tool["name"] == tool_name), None)
        if not tool_schema:
            logger.error(f"Tool {tool_name} not found")
            return f"Error: Tool {tool_name} not found"

        # Configure AutoGen LLM
        llm_config = {
            "config_list": [{
                "model": config.AUTOGEN_MODEL,
                "api_key": config.AUTOGEN_API_KEY,
                "base_url": config.AUTOGEN_API_BASE,
                "price": [0.0, 0.0],  # Dummy pricing to suppress cost warning
            }],
            "temperature": 0.3,
            "timeout": config.GEMMA_API_TIMEOUT,
            "max_tokens": 1000,
        }

        # Create AutoGen agents
        assistant = AssistantAgent(
            name="ToolAssistant",
            llm_config=llm_config,
            system_message="You are an expert assistant tasked with generating precise responses based on provided context and instructions. Provide only the requested output (e.g., summary, abstract, key findings, or challenges) as specified in the instructions. Do not include commentary, evaluations, or additional explanations beyond the required format."
        )

        user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,  # Single reply to avoid loops
            is_termination_msg=lambda x: x.get("content", "").strip() != "",  # Terminate after non-empty response
            code_execution_config=False,
        )

        # Prepare the prompt
        prompt = f"""You are tasked with executing the following tool:
{json.dumps(tool_schema, indent=2)}

Context:
{context}

Instructions:
{tool_schema['instructions']}

Additional Guidelines:
1. Base your response EXCLUSIVELY on the provided context.
2. Follow the structure and format specified in the instructions.
3. Ensure the response is concise, formal, and academic in tone.
4. If the context lacks sufficient information for a section, state: "Not explicitly mentioned in the provided context."
5. Do not invent information or include page references.
6. Provide only the requested output without commentary or evaluation."""

        # Log the prompt for debugging
        logger.debug(f"Prompt sent to assistant: {prompt[:500]}...")

        # Initiate the chat
        user_proxy.initiate_chat(
            assistant,
            message=prompt,
            clear_history=True
        )

        # Get the assistant's response (last message from assistant)
        last_message = assistant.last_message()
        logger.debug(f"Raw assistant response: {last_message}")

        # Check if response is valid
        response = last_message.get("content", "")
        if not response.strip():
            logger.error("Empty response received from assistant")
            return "Error: No response generated by the assistant"

        # Clean up response
        result = re.sub(r'\[Page \d+\]', '', response)  # Remove page references
        result = re.sub(r'\*\*([^\*]+)\*\*', r'\1', result)  # Remove bold
        result = re.sub(r'\*([^\*]+)\*', r'\1', result)  # Remove italics
        result = re.sub(r'^#+.*\n', '', result, flags=re.MULTILINE)  # Remove markdown headers
        result = re.sub(r'\n{3,}', '\n\n', result).strip()  # Normalize newlines

        # Validate response length for abstract (log only, no note)
        if tool_name == "abstract":
            word_count = len(result.split())
            if not (200 <= word_count <= 250):
                logger.warning(f"Abstract word count {word_count} outside 200-250 range")

        return result

    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
        return f"Error generating response: {str(e)}"