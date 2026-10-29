from langchain_core.prompts import ChatPromptTemplate

QA_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful FAQ assistant. Answer the question using ONLY the "
    "context below. Give a complete answer, formatted in Markdown for easy "
    "reading: start with a one or two sentence direct answer, then use short "
    "paragraphs, bullet lists, and **bold** key terms; use a `##` heading only "
    "when there are distinct sections. Never write one long wall of text. "
    "Cite the source file (and page, if given) at the end of each point "
    "like [source_file p.X]. If the answer isn't in the context, say you "
    "don't know rather than guessing.\n\n"
    "Conversation so far:\n{history}\n\nContext:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)

WEB_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "web search results below. Note these results may be time-sensitive. If "
    "the results don't contain the answer, say so.\n\n"
    "Conversation so far:\n{history}\n\nWeb Search Results:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Summarize the document content below in {mode} form.\n"
    "- executive: 3-5 sentences, high-level, for a decision-maker skimming it\n"
    "- detailed: thorough, covers all major points/findings/numbers, several paragraphs\n"
    "- bullet: concise bullet points, one per key idea\n\n"
    "If multiple files are included, summarize each under its own heading, "
    "then add a short combined takeaway.\n\n"
    "User's request: {question}\n\nDocument content:\n{context}\n\nSummary:"
)

COMPARE_PROMPT = ChatPromptTemplate.from_template(
    "Compare the documents below based on the user's request. Return a "
    "markdown comparison table where rows are the aspects being compared and "
    "columns are the documents, followed by a short prose summary of the key "
    "differences.\n\nUser's request: {question}\n\nDocuments:\n{context}\n\n"
    "Comparison:"
)

EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    "Extract the requested information from the content below and return it "
    "as a markdown table ONLY -- no prose before or after, just the table. "
    "Pick sensible column headers based on what the user asked for. If a "
    "field isn't present for a row, use an empty cell rather than guessing.\n\n"
    "User's request: {question}\n\nContent:\n{context}\n\nMarkdown table:"
)

CODE_GENERATE_PROMPT = ChatPromptTemplate.from_template(
    "Write code that satisfies this request. Include the language in a "
    "fenced code block, and a short explanation after the code (not before). "
    "Use any provided document context if relevant (e.g. implementing an "
    "algorithm described in a paper).\n\n"
    "Context (if relevant):\n{context}\n\nRequest: {question}\n\nCode:"
)

CODE_EXPLAIN_PROMPT = ChatPromptTemplate.from_template(
    "Explain what this code does, clearly and concisely. Cover: overall "
    "purpose, key steps, any non-obvious logic, and complexity if relevant.\n\n"
    "Code:\n{code}\n\nExplanation:"
)

CODE_DEBUG_PROMPT = ChatPromptTemplate.from_template(
    "Find and fix the bug(s) in this code. Explain what was wrong, then give "
    "the corrected code in a fenced code block.\n\n"
    "Code:\n{code}\n\nError/symptom (if provided): {error}\n\nDiagnosis and fix:"
)

CODE_OPTIMIZE_PROMPT = ChatPromptTemplate.from_template(
    "Optimize this code for {goal} (e.g. speed, memory, readability -- infer "
    "from context if not specified). Explain the key changes, then give the "
    "optimized code in a fenced code block.\n\nCode:\n{code}\n\nOptimization:"
)

IMAGE_DESCRIBE_PROMPT = (
    "Describe this image/figure/chart/diagram in detail. If it's a chart or "
    "graph, describe the axes, trend, and any notable data points. If it's a "
    "diagram, describe the components and how they relate. If it's a table "
    "rendered as an image, transcribe its content."
)

IMAGE_QA_PROMPT = (
    "Answer the question about this image/figure/chart/diagram as precisely "
    "as possible, based only on what's visible in it.\n\nQuestion: {question}"
)

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    "Classify this user request into exactly one category. Return ONLY the "
    "category name, nothing else.\n\n"
    "Categories:\n"
    "- qa: a question to be answered from documents or general knowledge (including explanations of concepts, theory, algorithms, or how things work)\n"
    "- summarize: asking for a summary/overview/recap of document(s)\n"
    "- compare: asking to compare two or more documents\n"
    "- table: asking to extract structured data into a table\n"
    "- code: asking to write, modify, debug, optimize, or explain specific programming code/scripts. Conceptual questions about algorithms or machine learning models belong in 'qa' unless the user explicitly asks for code implementation.\n"
    "- image: asking about a figure, chart, diagram, or image in a document -- descriptive/qualitative understanding\n"
    "- vision_analysis: asking to detect, count, or locate objects in an image -- quantitative/measurable output (bounding boxes, counts)\n\n"
    "Request: {question}\n\nCategory:"
)

PLANNER_PROMPT = ChatPromptTemplate.from_template(
    "You are a task planner for a multi-agent RAG system. Convert the user's request "
    "into an ordered execution plan. Use only these agent names: retrieve, qa, summarize, "
    "compare, table, code, image, vision_analysis, verify. Use dependencies when one step should wait for "
    "another. Format each line exactly as:\n"
    "step_id | agent | depends_on comma-separated or none | instruction\n\n"
    "If multiple analysis tasks can run after one retrieval step, reuse that retrieve step "
    "and leave those tasks independent so they can run in parallel.\n\n"
    "Request: {question}\n\nPlan:"
)

VERIFY_PROMPT = ChatPromptTemplate.from_template(
    "You are a fact-checker. Given the source context and a generated "
    "answer, identify any claims in the answer that are NOT supported by the "
    "context (potential hallucinations). Return a short bullet list of "
    "unsupported claims, or \"None\" if the answer is fully supported.\n\n"
    "Context:\n{context}\n\nAnswer:\n{answer}\n\nUnsupported claims:"
)

REFLECTION_PROMPT = ChatPromptTemplate.from_template(
    "You are a reflection agent. Improve the answer only if the retrieved context suggests "
    "a factual correction, stronger citation grounding, or a clearer structure. Keep the answer "
    "concise and do not introduce claims that are not supported by the context.\n\n"
    "Question:\n{question}\n\nCurrent answer:\n{answer}\n\nContext:\n{context}\n\nRevised answer:"
)
