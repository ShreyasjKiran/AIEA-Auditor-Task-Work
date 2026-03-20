"""
Symbol-LLM Reimplementation with LangChain + RAG
==================================================
Paper: "Symbol-LLM: Towards Foundational Symbol-centric Interface for LLMs"
       Xu et al., ACL 2024 | arXiv:2311.09278

This implements the Symbol-LLM approach using LangChain:
  1. Parses a Prolog knowledge base into documents for RAG
  2. Uses a vector store (Chroma) to retrieve relevant KB facts
  3. LLM translates natural language queries into symbolic form (FOL)
  4. LLM performs step-by-step logical inference over retrieved facts
  5. Outputs: True/False verdict + full inference trace

Usage:
    python symbol_llm_langchain.py                    # Run all test queries
    python symbol_llm_langchain.py --query "Is Josh Allen a quarterback?"
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROLOG KB PARSER — Parse .pl file into structured documents for RAG
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
    """Parse a Prolog .pl file into facts and rules."""
    facts = []
    rules = []
    text = Path(filepath).read_text()

    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('%'):
            continue

        if ':-' in line:
            # It's a rule
            head, body = line.split(':-', 1)
            rules.append(PrologRule(
                head=head.strip(),
                body=body.strip().rstrip('.'),
                raw=line
            ))
        elif line.endswith('.'):
            # It's a fact — skip if it contains Prolog variables (uppercase start)
            match = re.match(r'(\w+)\(([^)]+)\)', line.rstrip('.'))
            if match:
                pred = match.group(1)
                args = [a.strip() for a in match.group(2).split(',')]
                # Skip lines with variables (e.g., X, Team, Player) or wildcards (_)
                if any(a[0].isupper() or a == '_' for a in args):
                    continue
                facts.append(PrologFact(predicate=pred, args=args, raw=line))

    return facts, rules


def kb_to_documents(facts: List[PrologFact], rules: List[PrologRule]) -> List[Document]:
    """Convert parsed KB into LangChain Documents for the vector store.

    Groups facts by predicate/team for better retrieval, and includes
    each rule as its own document with a natural language description.
    """
    docs = []

    # Group player facts by team
    team_players: Dict[str, List[PrologFact]] = {}
    team_facts_list: List[PrologFact] = []
    for f in facts:
        if f.predicate == 'player':
            team = f.args[1]
            team_players.setdefault(team, []).append(f)
        elif f.predicate == 'team':
            team_facts_list.append(f)

    # One document per team's roster
    for team, players in team_players.items():
        lines = [f.raw for f in players]
        player_names = [p.args[0].replace('_', ' ').title() for p in players]
        content = (
            f"Team: {team}\n"
            f"Players: {', '.join(player_names)}\n"
            f"Prolog facts:\n" + '\n'.join(lines)
        )
        docs.append(Document(
            page_content=content,
            metadata={"type": "roster", "team": team}
        ))

    # Team conference/division info grouped by division
    div_teams: Dict[str, List[PrologFact]] = {}
    for f in team_facts_list:
        div_key = (f.args[1], f.args[2])  # e.g. ("afc", "east")
        div_teams.setdefault(div_key, []).append(f)

    for (conf, div), tfs in div_teams.items():
        team_names = [t.args[0] for t in tfs]
        content = (
            f"Division: {conf.upper()} {div.title()}\n"
            f"Teams: {', '.join(team_names)}\n"
            f"Prolog facts:\n" + '\n'.join(t.raw for t in tfs)
        )
        docs.append(Document(
            page_content=content,
            metadata={"type": "division", "conference": conf, "division": div}
        ))

    # Each rule as its own document with NL explanation
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
        content = (
            f"Rule: {name}\n"
            f"Description: {desc}\n"
            f"Prolog:\n{r.raw}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"type": "rule", "rule_name": name}
        ))

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# 2. RAG VECTOR STORE — Build and query the Chroma store
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorstore(docs: List[Document], persist_dir: str = "./chroma_db") -> Chroma:
    """Build a Chroma vector store from KB documents."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def get_retriever(vectorstore: Chroma, k: int = 6):
    """Get a retriever that fetches the top-k most relevant documents."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. SYMBOL-LLM LANGCHAIN — The core inference chain
# ─────────────────────────────────────────────────────────────────────────────

# Full FOL axioms for the system prompt (Symbol-LLM's symbolic interface)
FOL_AXIOMS = """\
% Player facts: PlayerFact(name, team, position) -> Player exists with those attributes
forall X T P: PlayerFact(X, T, P) -> Player(X) ^ PlaysFor(X, T) ^ HasPosition(X, P)

% Team facts: TeamFact(team, conference, division) -> Team exists with those attributes
forall T C D: TeamFact(T, C, D) -> Team(T) ^ InConference(T, C) ^ InDivision(T, D)

% Teammates rule
forall X Y T: PlaysFor(X, T) ^ PlaysFor(Y, T) ^ X!=Y -> Teammates(X, Y)

% Conference rule
forall P T C: PlaysFor(P, T) ^ InConference(T, C) -> PlaysInConference(P, C)

% Division rule
forall P T D: PlaysFor(P, T) ^ InDivision(T, D) -> PlaysInDivision(P, D)

% Division rivals rule
forall X Y Tx Ty D: PlaysFor(X,Tx) ^ PlaysFor(Y,Ty) ^ InDivision(Tx,D) ^ InDivision(Ty,D) ^ Tx!=Ty
    -> DivisionRivals(X, Y)

% Offense/defense rules
forall P: HasPosition(P, qb) v HasPosition(P, wr) v HasPosition(P, rb)
        v HasPosition(P, te) v HasPosition(P, ot) -> PlaysOffense(P)
forall P: HasPosition(P, de) v HasPosition(P, dt) v HasPosition(P, lb)
        v HasPosition(P, cb) v HasPosition(P, s)  -> PlaysDefense(P)
"""

SYSTEM_PROMPT = """\
You are Symbol-LLM, an expert in both natural language and formal symbolic reasoning systems \
(First-Order Logic, Prolog). You follow the Symbol-LLM methodology: given a natural language \
query and a knowledge base, you MUST:

1. TRANSLATE the natural language query into a formal First-Order Logic (FOL) expression.
2. RETRIEVE and list the specific facts from the knowledge base that are relevant.
3. APPLY the appropriate inference rules step-by-step, showing each deduction.
4. CONCLUDE with a definitive TRUE or FALSE verdict.

Here are the FOL axioms that define the inference rules:

{fol_axioms}

Your output MUST follow this exact format:

QUERY (FOL): <the FOL translation of the question>

RELEVANT FACTS:
- <fact 1>
- <fact 2>
...

INFERENCE TRACE:
Step 1: <first deduction with rule applied>
Step 2: <next deduction>
...

VERDICT: TRUE or FALSE
"""

HUMAN_PROMPT = """\
Knowledge Base Context (retrieved via RAG):
{context}

Question: {question}
"""


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_inference_chain(retriever, model_name: str = "gpt-4o-mini"):
    """Build the full Symbol-LLM LangChain inference chain.

    Pipeline: question -> RAG retrieval -> symbolic translation -> inference -> verdict
    """
    llm = ChatOpenAI(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "fol_axioms": lambda _: FOL_AXIOMS,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRUCTURED OUTPUT PARSER — Extract verdict and trace
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    question: str
    fol_query: str
    relevant_facts: List[str]
    inference_trace: List[str]
    verdict: bool
    raw_response: str


def parse_inference_response(question: str, response: str) -> InferenceResult:
    """Parse the structured LLM response into an InferenceResult."""
    # Extract FOL query
    fol_match = re.search(r'QUERY \(FOL\):\s*(.+)', response)
    fol_query = fol_match.group(1).strip() if fol_match else "N/A"

    # Extract relevant facts
    facts = []
    facts_section = re.search(r'RELEVANT FACTS:\n(.*?)(?=\nINFERENCE TRACE:)', response, re.DOTALL)
    if facts_section:
        for line in facts_section.group(1).strip().split('\n'):
            line = line.strip().lstrip('- ')
            if line:
                facts.append(line)

    # Extract inference trace
    trace = []
    trace_section = re.search(r'INFERENCE TRACE:\n(.*?)(?=\nVERDICT:)', response, re.DOTALL)
    if trace_section:
        for line in trace_section.group(1).strip().split('\n'):
            line = line.strip()
            if line:
                trace.append(line)

    # Extract verdict
    verdict_match = re.search(r'VERDICT:\s*(TRUE|FALSE)', response, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() == 'TRUE' if verdict_match else False

    return InferenceResult(
        question=question,
        fol_query=fol_query,
        relevant_facts=facts,
        inference_trace=trace,
        verdict=verdict,
        raw_response=response,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEST SUITE — Queries to evaluate the system
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
# 6. MAIN — Run inference and display results
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: InferenceResult):
    """Pretty-print an inference result with the full trace."""
    print(f"\n{'='*70}")
    print(f"QUESTION: {result.question}")
    print(f"{'='*70}")
    print(f"\nFOL Translation: {result.fol_query}")
    print(f"\nRelevant Facts Retrieved (via RAG):")
    for fact in result.relevant_facts:
        print(f"  - {fact}")
    print(f"\nInference Trace:")
    for step in result.inference_trace:
        print(f"  {step}")
    print(f"\n>>> VERDICT: {'TRUE' if result.verdict else 'FALSE'}")
    print(f"{'='*70}")


def run_single_query(chain, question: str) -> InferenceResult:
    """Run a single query through the chain and parse the result."""
    response = chain.invoke(question)
    return parse_inference_response(question, response)


def run_evaluation(chain, queries: List[Dict]) -> Dict:
    """Run all test queries and report accuracy."""
    correct = 0
    total = len(queries)
    results = []

    print(f"\n{'#'*70}")
    print(f"  SYMBOL-LLM + LANGCHAIN + RAG — NFL KB EVALUATION")
    print(f"  {total} test queries")
    print(f"{'#'*70}")

    for i, q in enumerate(queries, 1):
        print(f"\n[{i}/{total}] Processing: {q['question']}")
        result = run_single_query(chain, q["question"])
        print_result(result)

        is_correct = result.verdict == q["expected"]
        correct += int(is_correct)
        status = "CORRECT" if is_correct else "WRONG"
        expected_str = "TRUE" if q["expected"] else "FALSE"
        got_str = "TRUE" if result.verdict else "FALSE"
        print(f"  Expected: {expected_str} | Got: {got_str} | {status}")
        results.append({
            "question": q["question"],
            "expected": q["expected"],
            "got": result.verdict,
            "correct": is_correct,
        })

    # Summary
    print(f"\n{'#'*70}")
    print(f"  FINAL RESULTS: {correct}/{total} correct ({correct/total:.0%})")
    print(f"{'#'*70}\n")

    print(f"{'Question':<60} {'Expected':<10} {'Got':<10} {'OK'}")
    print(f"{'-'*60} {'-'*10} {'-'*10} {'-'*4}")
    for r in results:
        exp = "TRUE" if r["expected"] else "FALSE"
        got = "TRUE" if r["got"] else "FALSE"
        ok = "Y" if r["correct"] else "N"
        print(f"{r['question']:<60} {exp:<10} {got:<10} {ok}")

    return {"correct": correct, "total": total, "accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser(
        description="Symbol-LLM + LangChain + RAG — NFL KB Logical Inference Engine"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single natural language query to evaluate (e.g., 'Is Josh Allen a quarterback?')"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--kb", type=str, default=None,
        help="Path to Prolog knowledge base (default: nfl_kb.pl in same directory)"
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"] == "your-openai-api-key-here":
        print("ERROR: Set your OPENAI_API_KEY in the .env file.")
        print("  Edit task8/.env and replace 'your-openai-api-key-here' with your actual key.")
        return

    # Resolve KB path
    script_dir = Path(__file__).parent
    kb_path = args.kb or str(script_dir / "nfl_kb.pl")

    print("Symbol-LLM + LangChain + RAG")
    print("Paper: arXiv:2311.09278 | ACL 2024\n")

    # Step 1: Parse the Prolog KB
    print("[1/4] Parsing Prolog knowledge base...")
    facts, rules = parse_prolog_kb(kb_path)
    print(f"      Loaded {len(facts)} facts and {len(rules)} rules from {kb_path}")

    # Step 2: Build documents and vector store
    print("[2/4] Building RAG vector store (Chroma + OpenAI embeddings)...")
    docs = kb_to_documents(facts, rules)
    print(f"      Created {len(docs)} documents for retrieval")
    vectorstore = build_vectorstore(docs, persist_dir=str(script_dir / "chroma_db"))
    retriever = get_retriever(vectorstore, k=6)

    # Step 3: Build the inference chain
    print(f"[3/4] Building LangChain inference chain (model: {args.model})...")
    chain = build_inference_chain(retriever, model_name=args.model)

    # Step 4: Run queries
    print("[4/4] Running inference...\n")

    if args.query:
        # Single query mode
        result = run_single_query(chain, args.query)
        print_result(result)
    else:
        # Full evaluation mode
        run_evaluation(chain, TEST_QUERIES)


if __name__ == "__main__":
    main()
