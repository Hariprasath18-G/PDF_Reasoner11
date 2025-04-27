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
This study addresses the challenge of efficient sequence transduction in neural machine translation, where traditional models struggle with long-range dependencies. We propose a transformer architecture that leverages multi-head self-attention to capture contextual relationships across input sequences. The methodology involves training a six-layer encoder-decoder model on a multilingual corpus, optimizing for BLEU scores. Results demonstrate a 15% improvement in translation accuracy over recurrent neural networks, with reduced training time. The transformer's ability to parallelize computations enhances scalability, making it suitable for large-scale applications. This work contributes to advancing machine translation systems, offering a robust framework for future research in natural language processing. Limitations include high computational requirements, suggesting avenues for optimizing resource efficiency."""
    },
    {
        "name": "key_findings",
        "description": "Extract and list significant qualitative insights and novel contributions from the context.",
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
        "instructions": """**Comprehensive Key Findings Extraction Guidelines**:

1. **Qualitative Insights**:
   - Identify important observations, trends, patterns, or conclusions
   - Include statements about effectiveness, efficiency, or impact
   - Extract insights about methodology, data, or theoretical contributions
   - Look for statements about implications for practice or policy
   - Example: "Early intervention showed benefits across all outcomes"

2. **Novel Contributions**:
   - Identify any new methods, models, approaches, or techniques
   - Highlight unique aspects of the research or innovative applications
   - Note any new frameworks, systems, or tools developed
   - Include any significant extensions of existing work
   - Example: "Introduced a novel ensemble method combining X and Y"

**Output Format Requirements**:
- Use clear bullet points under each category
- Never use "Not mentioned" - instead try to infer from context
- Be as comprehensive as possible without inventing information
- Maintain original meaning but paraphrase concisely
- Remove any citations or page references

Example Output:
- **Qualitative Insights**:
  - Feature importance analysis revealed unexpected patterns
  - Method demonstrated robustness across diverse datasets
- **Novel Contributions**:
  - Developed new interpretability framework for model decisions
  - Introduced hybrid architecture combining X and Y approaches"""
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

        # Configure AutoGen LLM - use higher temp for key findings
        temperature = 0.5 if tool_name == "key_findings" else 0.3
        max_tokens = 1500 if tool_name == "key_findings" else 1000
        
        llm_config = {
            "config_list": [{
                "model": config.AUTOGEN_MODEL,
                "api_key": config.AUTOGEN_API_KEY,
                "base_url": config.AUTOGEN_API_BASE,
                "price": [0.0, 0.0],
            }],
            "temperature": temperature,
            "timeout": config.GEMMA_API_TIMEOUT,
            "max_tokens": max_tokens,
        }

        # Create AutoGen agents
        assistant = AssistantAgent(
            name="ToolAssistant",
            llm_config=llm_config,
            system_message="You are an expert assistant tasked with generating precise responses based on provided context and instructions. Provide only the requested output as specified in the instructions. Do not include commentary or evaluations."
        )

        user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: x.get("content", "").strip() != "",
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
1. Base your response EXCLUSIVELY on the provided context. Do not include findings from unrelated topics or previous contexts.
2. Follow the exact structure and format specified in the instructions, using '- **Section Name**:' for headers and '- ' for bullet points.
3. Ensure the response is concise, formal, and academic in tone.
4. Do not invent information or include page references or citations.
5. Provide only the requested output without commentary or evaluation.
6. For the key_findings tool:
   - **Qualitative Insights**: Capture observations, trends, patterns, or implications, including methodology impacts and practical applications.
   - **Novel Contributions**: Highlight new methods, systems, or applications, emphasizing unique aspects or extensions of prior work.
7. Ensure each section has at least one bullet point, using inferences only when explicit findings are absent.
8. Avoid duplicating sections or including irrelevant findings (e.g., Alzheimer’s disease in a network security context)."""

        # Log the prompt for debugging
        logger.debug(f"Prompt sent to assistant: {prompt[:500]}...")

        # Initiate the chat
        user_proxy.initiate_chat(
            assistant,
            message=prompt,
            clear_history=True
        )

        # Get the assistant's response
        last_message = assistant.last_message()
        logger.debug(f"Raw assistant response: {last_message}")

        # Check if response is valid
        response = last_message.get("content", "")
        if not response.strip():
            logger.error("Empty response received from assistant")
            return "Error: No response generated by the assistant"

        # Enhanced cleaning for key findings
        if tool_name == "key_findings":
            response = _clean_key_findings(response, context)

        # Standard cleaning for all tools
        result = re.sub(r'\[Page \d+\]', '', response)  # Remove page references
        result = re.sub(r'\*\*([^\*]+)\*\*', r'\1', result)  # Remove bold
        result = re.sub(r'\*([^\*]+)\*', r'\1', result)  # Remove italics
        result = re.sub(r'^#+.*\n', '', result, flags=re.MULTILINE)  # Remove markdown headers
        result = re.sub(r'\n{3,}', '\n\n', result).strip()  # Normalize newlines

        # Validate response length for abstract
        if tool_name == "abstract":
            word_count = len(result.split())
            if not (200 <= word_count <= 250):
                logger.warning(f"Abstract word count {word_count} outside 200-250 range")

        return result

    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
        return f"Error generating response: {str(e)}"

def _clean_key_findings(response: str, context: str) -> str:
    """Special cleaning for key findings output, focusing on Qualitative Insights and Novel Contributions."""
    sections = ["Qualitative Insights", "Novel Contributions"]
    cleaned_lines = []
    seen_sections = set()
    findings_by_section = {section: [] for section in sections}

    # Normalize response: standardize headers, bullets, and remove extra whitespace
    response = re.sub(r'#{1,}\s*([^\n]+)', r'- **\1**:', response)  # Convert markdown headers
    response = re.sub(r'(\n\s*[-*]\s*|\n\s*)\*+', r'\n- ', response)  # Normalize bullets to '- '
    response = re.sub(r'\n\s*\n+', '\n', response)  # Remove extra newlines
    response = re.sub(r'\[Page \d+\]', '', response)  # Remove page references

    # Log normalized response for debugging
    logger.debug(f"Normalized response: {response[:500]}...")

    # Context keywords for relevance check
    context_lower = context.lower()
    context_keywords = ['ddos', 'network', 'big data', 'apache spark'] if 'ddos' in context_lower else ['research', 'study', 'model', 'data']

    # Split response into lines and process
    lines = response.split('\n')
    current_section = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if line is a section header
        section_match = re.match(r'-\s*\*\*([^\*]+)\*\*:', line)
        if section_match:
            section = section_match.group(1).strip()
            if section in sections:
                current_section = section
                if section not in seen_sections:
                    seen_sections.add(section)
                i += 1
                continue
        # Collect findings under current section
        if current_section and line.startswith('- '):
            finding = line[2:].strip()
            if finding and finding.lower() not in ['none', 'not mentioned', '']:
                # Validate finding relevance
                finding_lower = finding.lower()
                is_relevant = any(keyword in finding_lower for keyword in context_keywords) or not any(
                    irrelevant in finding_lower for irrelevant in ['alzheimer', 'disease', 'cognitive'] if 'ddos' in context_lower
                )
                if is_relevant:
                    findings_by_section[current_section].append(f"- {finding}")
        i += 1

    # Compile output, ensuring all sections are present
    for section in sections:
        cleaned_lines.append(f"- **{section}**:")
        findings = findings_by_section[section]
        if not findings:
            if section == "Qualitative Insights":
                cleaned_lines.append("- Inferred insights suggest the research provides valuable observations or practical implications aligned with its objectives.")
            else:  # Novel Contributions
                cleaned_lines.append("- Inferred contributions suggest the research introduces novel methods or applications relevant to its field.")
        else:
            # Remove duplicate findings within the same section
            unique_findings = []
            seen_findings = set()
            for finding in findings:
                if finding not in seen_findings:
                    unique_findings.append(finding)
                    seen_findings.add(finding)
            cleaned_lines.extend(unique_findings)

    # Log final output for debugging
    final_output = '\n'.join(cleaned_lines)
    logger.debug(f"Final cleaned output: {final_output[:500]}...")
    return final_output
