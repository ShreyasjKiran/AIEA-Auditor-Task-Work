"""
Symbol-LLM Reimplementation with LangGraph + RAG
==================================================
Paper: "Symbol-LLM: Towards Foundational Symbol-centric Interface for LLMs"
       Xu et al., ACL 2024 | arXiv:2311.09278

Migrated from Task 8's LangChain LCEL chain to a LangGraph StateGraph.
Key improvements over Task 8:
  - Explicit state management via TypedDict
  - Relevancy judge node: after RAG retrieval, LLM judges if docs are sufficient
  - Conditional routing: if docs are insufficient, re-retrieves with a refined query
  - Chain-of-thought self-refinement: reviews inference before finalizing verdict

Graph:
  retrieve -> judge_relevancy --(relevant)--> infer -> refine -> END
                               \--(not relevant)--> re_retrieve -> infer -> refine -> END

Usage:
    python symbol_llm_langgraph.py
    python symbol_llm_langgraph.py --query "Is Josh Allen a quarterback?"
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Annotated
from dataclasses import dataclass

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROLOG KB PARSER (reused from Task 8)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrologFact:
    predicate: str
    args: List[str]
    raw: str

@dataclass
class PrologRule:
    head: str
    body: str
    raw: str


def parse_prolog_kb(filepath: str) -> tuple[List[PrologFact], List[PrologRule]]:
    facts, rules = [], []
    text = Path(filepath).read_text()
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if ':-' in line:
            head, body = line.split(':-', 1)
            rules.append(PrologRule(head=head.strip(), body=body.strip().rstrip('.'), raw=line))
        elif line.endswith('.'):
            match = re.match(r'(\w+)\(([^)]+)\)', line.rstrip('.'))
            if match:
                pred = match.group(1)
                args = [a.strip() for a in match.group(2).split(',')]
                if any(a[0].isupper() or a == '_' for a in args):
                    continue
                facts.append(PrologFact(predicate=pred, args=args, raw=line))
    return facts, rules


def kb_to_documents(facts: List[PrologFact], rules: List[PrologRule]) -> List[Document]:
    docs = []

    # Group player facts by team
    team_players: Dict[str, List[PrologFact]] = {}
    team_facts_list: List[PrologFact] = []
    for f in facts:
        if f.predicate == 'player':
            team_players.setdefault(f.args[1], []).append(f)
        elif f.predicate == 'team':
            team_facts_list.append(f)

    for team, players in team_players.items():
        player_names = [p.args[0].replace('_', ' ').title() for p in players]
        content = (
            f"Team: {team}\n"
            f"Players: {', '.join(player_names)}\n"
            f"Prolog facts:\n" + '\n'.join(f.raw for f in players)
        )
        docs.append(Document(page_content=content, metadata={"type": "roster", "team": team}))

    # Division groupings
    div_teams: Dict[tuple, List[PrologFact]] = {}
    for f in team_facts_list:
        div_teams.setdefault((f.args[1], f.args[2]), []).append(f)

    for (conf, div), tfs in div_teams.items():
        team_names = [t.args[0] for t in tfs]
        content = (
            f"Division: {conf.upper()} {div.title()}\n"
            f"Teams: {', '.join(team_names)}\n"
            f"Prolog facts:\n" + '\n'.join(t.raw for t in tfs)
        )
        docs.append(Document(page_content=content, metadata={"type": "division", "conference": conf, "division": div}))

    # Rules
    rule_descriptions = {
        "teammates": "Two players are teammates if they play for the same team.",
        "plays_in_conference": "A player plays in a conference based on their team's conference.",
        "plays_in_division": "A player plays in a division based on their team's division.",
        "division_rivals": "Two players are division rivals if they are in the same division but on different teams.",
        "plays_offense": "A player plays offense if their position is one of: qb, wr, rb, te, ot, og, c.",
        "plays_defense": "A player plays defense if their position is one of: de, dt, lb, cb, s.",
        "team_roster": "Lists all players on a given team.",
    }
    for r in rules:
        rule_name = re.match(r'(\w+)', r.head)
        name = rule_name.group(1) if rule_name else "unknown"
        desc = rule_descriptions.get(name, "")
        content = f"Rule: {name}\nDescription: {desc}\nProlog:\n{r.raw}"
        docs.append(Document(page_content=content, metadata={"type": "rule", "rule_name": name}))

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# 2. VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorstore(docs: List[Document], persist_dir: str = "./chroma_db") -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FOL AXIOMS & PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

FOL_AXIOMS = """\
forall X T P: PlayerFact(X, T, P) -> Player(X) ^ PlaysFor(X, T) ^ HasPosition(X, P)
forall T C D: TeamFact(T, C, D) -> Team(T) ^ InConference(T, C) ^ InDivision(T, D)
forall X Y T: PlaysFor(X, T) ^ PlaysFor(Y, T) ^ X!=Y -> Teammates(X, Y)
forall P T C: PlaysFor(P, T) ^ InConference(T, C) -> PlaysInConference(P, C)
forall P T D: PlaysFor(P, T) ^ InDivision(T, D) -> PlaysInDivision(P, D)
forall X Y Tx Ty D: PlaysFor(X,Tx) ^ PlaysFor(Y,Ty) ^ InDivision(Tx,D) ^ InDivision(Ty,D) ^ Tx!=Ty -> DivisionRivals(X, Y)
forall P: HasPosition(P, qb) v HasPosition(P, wr) v HasPosition(P, rb) v HasPosition(P, te) v HasPosition(P, ot) -> PlaysOffense(P)
forall P: HasPosition(P, de) v HasPosition(P, dt) v HasPosition(P, lb) v HasPosition(P, cb) v HasPosition(P, s) -> PlaysDefense(P)
"""

RELEVANCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevancy judge. Given a question and retrieved knowledge base documents, "
     "determine if the documents contain ALL the facts needed to answer the question.\n\n"
     "Respond with EXACTLY one of:\n"
     "RELEVANT — if the documents contain sufficient facts\n"
     "NOT_RELEVANT — if critical facts are missing\n\n"
     "Then on the next line, briefly explain what is present or missing. "
     "If NOT_RELEVANT, on a third line write REFINED_QUERY: <a better search query to find the missing facts>"),
    ("human",
     "Question: {question}\n\nRetrieved Documents:\n{context}"),
])

INFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are Symbol-LLM, an expert in formal symbolic reasoning (FOL, Prolog). "
     "Given a question and knowledge base facts, you MUST:\n\n"
     "1. TRANSLATE the question into First-Order Logic (FOL)\n"
     "2. LIST the specific relevant facts from the KB\n"
     "3. APPLY inference rules step-by-step (show each deduction)\n"
     "4. CONCLUDE with TRUE or FALSE\n\n"
     "FOL Axioms:\n{fol_axioms}\n\n"
     "Output format:\n"
     "QUERY (FOL): <FOL expression>\n\n"
     "RELEVANT FACTS:\n- <fact>\n...\n\n"
     "INFERENCE TRACE:\n"
     "Step 1: <deduction>\n...\n\n"
     "VERDICT: TRUE or FALSE"),
    ("human",
     "Knowledge Base Context:\n{context}\n\nQuestion: {question}"),
])

REFINEMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a logical reasoning reviewer. You are given a question, knowledge base facts, "
     "and a draft inference with a verdict. Your job is to:\n\n"
     "1. CHECK if the inference trace is logically sound — does each step follow from the previous?\n"
     "2. CHECK if the correct rules were applied\n"
     "3. CHECK if the verdict matches the trace\n"
     "4. If there are errors, CORRECT them\n\n"
     "FOL Axioms:\n{fol_axioms}\n\n"
     "Output the FINAL corrected result in this format:\n"
     "REVIEW: <brief assessment — SOUND or CORRECTED>\n\n"
     "QUERY (FOL): <FOL expression>\n\n"
     "RELEVANT FACTS:\n- <fact>\n...\n\n"
     "INFERENCE TRACE:\n"
     "Step 1: <deduction>\n...\n\n"
     "VERDICT: TRUE or FALSE"),
    ("human",
     "Question: {question}\n\nKB Context:\n{context}\n\nDraft Inference:\n{draft_inference}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────

from typing import TypedDict


class GraphState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    context: str
    relevancy: str
    refined_query: str
    retrieval_count: int
    draft_inference: str
    final_inference: str
    review_status: str
    verdict: bool
    fol_query: str
    relevant_facts: List[str]
    inference_trace: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# 5. GRAPH NODES
# ─────────────────────────────────────────────────────────────────────────────

def make_nodes(retriever, llm):
    """Create all graph node functions, closed over the retriever and LLM."""

    def retrieve(state: GraphState) -> GraphState:
        """Retrieve relevant documents from the vector store."""
        query = state.get("refined_query") or state["question"]
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {
            **state,
            "retrieved_docs": docs,
            "context": context,
            "retrieval_count": state.get("retrieval_count", 0) + 1,
        }

    def judge_relevancy(state: GraphState) -> GraphState:
        """LLM judges whether retrieved docs are sufficient to answer the question."""
        response = (RELEVANCY_PROMPT | llm).invoke({
            "question": state["question"],
            "context": state["context"],
        })
        text = response.content

        if "NOT_RELEVANT" in text:
            relevancy = "NOT_RELEVANT"
            refined_match = re.search(r'REFINED_QUERY:\s*(.+)', text)
            refined_query = refined_match.group(1).strip() if refined_match else state["question"]
        else:
            relevancy = "RELEVANT"
            refined_query = state.get("refined_query", "")

        return {
            **state,
            "relevancy": relevancy,
            "refined_query": refined_query,
        }

    def infer(state: GraphState) -> GraphState:
        """Perform Symbol-LLM inference: translate to FOL and reason step-by-step."""
        response = (INFERENCE_PROMPT | llm).invoke({
            "question": state["question"],
            "context": state["context"],
            "fol_axioms": FOL_AXIOMS,
        })
        return {**state, "draft_inference": response.content}

    def refine(state: GraphState) -> GraphState:
        """Self-refinement: review the draft inference for logical soundness."""
        response = (REFINEMENT_PROMPT | llm).invoke({
            "question": state["question"],
            "context": state["context"],
            "draft_inference": state["draft_inference"],
            "fol_axioms": FOL_AXIOMS,
        })
        text = response.content

        # Parse the final refined output
        review_match = re.search(r'REVIEW:\s*(.+)', text)
        review_status = review_match.group(1).strip() if review_match else "UNKNOWN"

        fol_match = re.search(r'QUERY \(FOL\):\s*(.+)', text)
        fol_query = fol_match.group(1).strip() if fol_match else "N/A"

        facts = []
        facts_section = re.search(r'RELEVANT FACTS:\n(.*?)(?=\nINFERENCE TRACE:)', text, re.DOTALL)
        if facts_section:
            for line in facts_section.group(1).strip().split('\n'):
                line = line.strip().lstrip('- ')
                if line:
                    facts.append(line)

        trace = []
        trace_section = re.search(r'INFERENCE TRACE:\n(.*?)(?=\nVERDICT:)', text, re.DOTALL)
        if trace_section:
            for line in trace_section.group(1).strip().split('\n'):
                line = line.strip()
                if line:
                    trace.append(line)

        verdict_match = re.search(r'VERDICT:\s*(TRUE|FALSE)', text, re.IGNORECASE)
        verdict = verdict_match.group(1).upper() == 'TRUE' if verdict_match else False

        return {
            **state,
            "final_inference": text,
            "review_status": review_status,
            "verdict": verdict,
            "fol_query": fol_query,
            "relevant_facts": facts,
            "inference_trace": trace,
        }

    return retrieve, judge_relevancy, infer, refine


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONDITIONAL ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def should_re_retrieve(state: GraphState) -> str:
    """Route based on relevancy judgment. Re-retrieve at most once."""
    if state["relevancy"] == "NOT_RELEVANT" and state.get("retrieval_count", 1) < 2:
        return "re_retrieve"
    return "infer"


# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(retriever, model_name: str = "gpt-4o-mini"):
    """
    Build the LangGraph StateGraph:

    START -> retrieve -> judge_relevancy --RELEVANT--> infer -> refine -> END
                                          \--NOT_RELEVANT--> re_retrieve -> infer -> refine -> END
    """
    llm = ChatOpenAI(model=model_name, temperature=0)
    retrieve, judge_relevancy, infer, refine = make_nodes(retriever, llm)

    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("judge_relevancy", judge_relevancy)
    graph.add_node("re_retrieve", retrieve)  # same function, different node name
    graph.add_node("infer", infer)
    graph.add_node("refine", refine)

    # Add edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "judge_relevancy")
    graph.add_conditional_edges("judge_relevancy", should_re_retrieve, {
        "re_retrieve": "re_retrieve",
        "infer": "infer",
    })
    graph.add_edge("re_retrieve", "infer")
    graph.add_edge("infer", "refine")
    graph.add_edge("refine", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 8. RESULT DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_result(state: GraphState):
    print(f"\n{'='*70}")
    print(f"QUESTION: {state['question']}")
    print(f"{'='*70}")
    print(f"\nRetrievals: {state.get('retrieval_count', 1)}")
    print(f"Relevancy: {state.get('relevancy', 'N/A')}")
    if state.get("refined_query"):
        print(f"Refined Query: {state['refined_query']}")
    print(f"Review: {state.get('review_status', 'N/A')}")
    print(f"\nFOL Translation: {state.get('fol_query', 'N/A')}")
    print(f"\nRelevant Facts (from KB via RAG):")
    for fact in state.get("relevant_facts", []):
        print(f"  - {fact}")
    print(f"\nInference Trace:")
    for step in state.get("inference_trace", []):
        print(f"  {step}")
    print(f"\n>>> VERDICT: {'TRUE' if state.get('verdict') else 'FALSE'}")
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    {"question": "Are Joe Burrow and Ja'Marr Chase teammates?", "expected": True},
    {"question": "Are Patrick Mahomes and Josh Allen teammates?", "expected": False},
    {"question": "Does Josh Allen play in the AFC?", "expected": True},
    {"question": "Does Jalen Hurts play in the NFC?", "expected": True},
    {"question": "Does Patrick Mahomes play in the NFC?", "expected": False},
    {"question": "Does Lamar Jackson play in the North division?", "expected": True},
    {"question": "Are Josh Allen and Drake Maye division rivals?", "expected": True},
    {"question": "Are Patrick Mahomes and Lamar Jackson division rivals?", "expected": False},
    {"question": "Does Saquon Barkley play offense?", "expected": True},
    {"question": "Does Myles Garrett play offense?", "expected": False},
    {"question": "Does Nick Bosa play defense?", "expected": True},
    {"question": "Does Brock Purdy play defense?", "expected": False},
    {"question": "Are Caleb Williams and Luther Burden teammates?", "expected": True},
    {"question": "Are Jalen Hurts and Jayden Daniels division rivals?", "expected": True},
    {"question": "Does Kyle Hamilton play defense?", "expected": True},
    {"question": "Does Tyreek Hill play in the East division?", "expected": True},
]


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Symbol-LLM + LangGraph + RAG — NFL KB Logical Inference Engine"
    )
    parser.add_argument("--query", type=str, default=None,
        help="Single natural language query")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
        help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--kb", type=str, default=None,
        help="Path to Prolog KB (default: nfl_kb.pl)")
    args = parser.parse_args()

    # Load .env
    script_dir = Path(__file__).parent
    load_dotenv(script_dir / ".env")
    if not os.environ.get("OPENAI_API_KEY") or "your-openai" in os.environ.get("OPENAI_API_KEY", ""):
        print("ERROR: Set your OPENAI_API_KEY in task9/.env")
        return

    kb_path = args.kb or str(script_dir / "nfl_kb.pl")

    print("Symbol-LLM + LangGraph + RAG")
    print("Paper: arXiv:2311.09278 | ACL 2024\n")

    # Step 1: Parse KB
    print("[1/4] Parsing Prolog knowledge base...")
    facts, rules = parse_prolog_kb(kb_path)
    print(f"      {len(facts)} facts, {len(rules)} rules")

    # Step 2: Build vector store
    print("[2/4] Building RAG vector store...")
    docs = kb_to_documents(facts, rules)
    print(f"      {len(docs)} documents")
    vectorstore = build_vectorstore(docs, persist_dir=str(script_dir / "chroma_db"))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Step 3: Build the graph
    print(f"[3/4] Building LangGraph (model: {args.model})...")
    app = build_graph(retriever, model_name=args.model)

    # Step 4: Run
    print("[4/4] Running inference...\n")

    if args.query:
        result = app.invoke({"question": args.query, "retrieval_count": 0})
        print_result(result)
    else:
        # Full evaluation
        correct = 0
        total = len(TEST_QUERIES)
        results = []

        print(f"{'#'*70}")
        print(f"  SYMBOL-LLM + LANGGRAPH + RAG — NFL KB EVALUATION ({total} queries)")
        print(f"{'#'*70}")

        for i, q in enumerate(TEST_QUERIES, 1):
            print(f"\n[{i}/{total}] {q['question']}")
            result = app.invoke({"question": q["question"], "retrieval_count": 0})
            print_result(result)

            is_correct = result.get("verdict") == q["expected"]
            correct += int(is_correct)
            exp = "TRUE" if q["expected"] else "FALSE"
            got = "TRUE" if result.get("verdict") else "FALSE"
            print(f"  Expected: {exp} | Got: {got} | {'CORRECT' if is_correct else 'WRONG'}")
            results.append({"question": q["question"], "expected": q["expected"],
                            "got": result.get("verdict"), "correct": is_correct,
                            "retrievals": result.get("retrieval_count", 1),
                            "review": result.get("review_status", "")})

        # Summary
        print(f"\n{'#'*70}")
        print(f"  FINAL: {correct}/{total} ({correct/total:.0%})")
        print(f"{'#'*70}\n")

        re_retrieved = sum(1 for r in results if r["retrievals"] > 1)
        corrected = sum(1 for r in results if "CORRECTED" in r.get("review", "").upper())
        print(f"Re-retrievals triggered: {re_retrieved}/{total}")
        print(f"Self-corrections applied: {corrected}/{total}\n")

        print(f"{'Question':<55} {'Exp':<6} {'Got':<6} {'Ret':<4} {'OK'}")
        print(f"{'-'*55} {'-'*6} {'-'*6} {'-'*4} {'-'*3}")
        for r in results:
            exp = "TRUE" if r["expected"] else "FALSE"
            got = "TRUE" if r["got"] else "FALSE"
            ok = "Y" if r["correct"] else "N"
            print(f"{r['question']:<55} {exp:<6} {got:<6} {r['retrievals']:<4} {ok}")


if __name__ == "__main__":
    main()
