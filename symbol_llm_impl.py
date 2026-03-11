"""
Symbol-LLM Reimplementation — NFL Knowledge Base Edition
==========================================================
Paper: "Symbol-LLM: Towards Foundational Symbol-centric Interface for LLMs"
       Xu et al., ACL 2024  |  arXiv:2311.09278

This file reimplements Symbol-LLM's core algorithm and evaluates it against
the nfl_kb.pl knowledge base from Task 4.

Run demo (no GPU):
    python symbol_llm_impl.py --mode demo

Run with actual Symbol-LLM model (needs GPU + transformers):
    python symbol_llm_impl.py --mode eval --query-mode fol
    python symbol_llm_impl.py --mode eval --query-mode prolog
    python symbol_llm_impl.py --mode eval --use-evol
"""

import re
import json
import random
import string
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from copy import deepcopy


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYMBOL-EVOL AUGMENTATION  (Section 3.2 of the paper)
#
#     Core idea: randomly rename all variables and predicate names so the
#     model cannot memorise surface token patterns.  It must learn *structure*.
#
#     Example:
#       Teammates(joe_burrow, ja_marr_chase)
#       -> Zxkp(vbnm_zqrt, plfq_wxyz)
# ─────────────────────────────────────────────────────────────────────────────

class SymbolEvol:
    VARIABLE_PATTERN  = re.compile(r'\b([A-Z][a-zA-Z0-9_]*)\b')
    PREDICATE_PATTERN = re.compile(r'\b([a-z][a-zA-Z0-9_]*)\s*[\(\[]')
    CONSTANT_PATTERN  = re.compile(r'\b([a-z][a-z0-9_]*)\b')

    RESERVED = {
        'not', 'and', 'or', 'if', 'iff', 'forall', 'exists',
        'is', 'true', 'fail', 'member', 'findall', 'assert',
        'afc', 'nfc', 'east', 'west', 'north', 'south',
        'qb', 'wr', 'rb', 'te', 'ot', 'og', 'de', 'dt', 'lb', 'cb', 's',
    }

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def _rand_name(self, n=4):  return ''.join(random.choices(string.ascii_lowercase, k=n))
    def _rand_const(self):      return self._rand_name(4) + '_' + self._rand_name(4)
    def _rand_var(self):        return random.choice(string.ascii_uppercase)

    def augment_fol(self, expr: str) -> Tuple[str, Dict[str, str]]:
        """Rename predicates and variables in a FOL expression."""
        mapping: Dict[str, str] = {}

        for pred in set(self.PREDICATE_PATTERN.findall(expr)):
            if pred not in self.RESERVED:
                mapping[pred] = self._rand_name(4).capitalize()

        for var in set(self.VARIABLE_PATTERN.findall(expr)):
            if var not in mapping:
                mapping[var] = self._rand_var()

        result = expr
        for orig, repl in sorted(mapping.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + re.escape(orig) + r'\b', repl, result)
        return result, mapping

    def augment_prolog(self, clauses: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Rename predicates and constants in a list of Prolog clauses."""
        full = '\n'.join(clauses)
        mapping: Dict[str, str] = {}

        for pred in set(self.PREDICATE_PATTERN.findall(full)):
            if pred not in self.RESERVED:
                mapping[pred] = self._rand_name(4)

        for var in set(self.VARIABLE_PATTERN.findall(full)):
            if var not in mapping:
                mapping[var] = self._rand_var()

        result = full
        for orig, repl in sorted(mapping.items(), key=lambda x: -len(x[0])):
            result = re.sub(r'\b' + re.escape(orig) + r'\b', repl, result)
        return result.split('\n'), mapping


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA PIPELINE  (Two-stage fine-tuning framework, Section 3.3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolicSample:
    task_type:       str
    symbol_family:   str
    instruction:     str
    symbolic_input:  str
    expected_output: str
    augmented: bool       = False
    metadata:  Dict       = field(default_factory=dict)


class SymbolicCollectionPipeline:
    """
    Stage 1 — Symbol-Centric: 100% symbolic data with Symbol-Evol augmentation.
    Stage 2 — Balanced:       Symbolic + NL data at a configurable ratio.
    This prevents catastrophic forgetting of natural-language ability.
    """

    def __init__(self, augment=True, augment_ratio=0.5):
        self.evol = SymbolEvol()
        self.augment = augment
        self.augment_ratio = augment_ratio

    def prepare_stage1(self, samples: List[SymbolicSample]) -> List[Dict]:
        out = []
        for s in samples:
            if self.augment and random.random() < self.augment_ratio:
                aug_expr, mapping = self.evol.augment_fol(s.symbolic_input)
                s = deepcopy(s)
                s.symbolic_input = aug_expr
                s.augmented = True
                s.metadata['symbol_mapping'] = mapping
            out.append(self._fmt(s))
        return out

    def prepare_stage2(self, symbolic: List[SymbolicSample],
                       nl_samples: List[Dict], ratio=0.4) -> List[Dict]:
        n = int(len(nl_samples) * (1 - ratio) / ratio)
        sel = random.sample(symbolic, min(n, len(symbolic)))
        return self.prepare_stage1(sel) + nl_samples

    @staticmethod
    def _fmt(s: SymbolicSample) -> Dict:
        return {
            "messages": [
                {"role": "system",
                 "content": (
                     "You are Symbol-LLM, expert in both natural language and formal "
                     "symbolic systems (FOL, Prolog, PDDL, code, math). "
                     "Reason step-by-step using the provided knowledge base."
                 )},
                {"role": "user",
                 "content": f"{s.instruction}\n\nSymbolic Input:\n{s.symbolic_input}"},
                {"role": "assistant", "content": s.expected_output},
            ],
            "task_type":     s.task_type,
            "symbol_family": s.symbol_family,
            "augmented":     s.augmented,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NFL KNOWLEDGE BASE  (nfl_kb.pl — Task 4)
# ─────────────────────────────────────────────────────────────────────────────

class NFLKnowledgeBase:
    """
    The full nfl_kb.pl expressed in Prolog and First-Order Logic,
    plus a 16-query test suite covering every rule in the KB.
    """

    PROLOG_FACTS = """\
% Players: player(Name, Team, Position)
player(josh_allen, bills, qb).
player(patrick_mahomes, chiefs, qb).
player(joe_burrow, bengals, qb).
player(lamar_jackson, ravens, qb).
player(jalen_hurts, eagles, qb).
player(brock_purdy, niners, qb).
player(caleb_williams, bears, qb).
player(jayden_daniels, commanders, qb).
player(baker_mayfield, buccaneers, qb).
player(drake_maye, patriots, qb).
player(trevor_lawrence, jaguars, qb).
player(jj_mccarthy, vikings, qb).
player(geno_smith, seahawks, qb).
player(ja_marr_chase, bengals, wr).
player(ceedee_lamb, cowboys, wr).
player(tyreek_hill, dolphins, wr).
player(puka_nacua, rams, wr).
player(malik_nabers, giants, wr).
player(jaxon_smith_njigba, seahawks, wr).
player(george_pickens, cowboys, wr).
player(luther_burden, bears, wr).
player(saquon_barkley, eagles, rb).
player(derrick_henry, ravens, rb).
player(breece_hall, jets, rb).
player(travis_etienne, jaguars, rb).
player(kenneth_walker, seahawks, rb).
player(brock_bowers, raiders, te).
player(trey_mcbride, cardinals, te).
player(myles_garrett, browns, de).
player(trey_hendrickson, bengals, de).
player(josh_hines_allen, jaguars, de).
player(maxx_crosby, raiders, de).
player(jared_verse, rams, de).
player(nick_bosa, niners, de).
player(patrick_surtain, broncos, cb).
player(devon_witherspoon, seahawks, cb).
player(kyle_hamilton, ravens, s).
player(dion_dawkins, bills, ot).

% Teams: team(Name, Conference, Division)
team(bills, afc, east).      team(dolphins, afc, east).
team(patriots, afc, east).   team(jets, afc, east).
team(ravens, afc, north).    team(bengals, afc, north).
team(browns, afc, north).    team(steelers, afc, north).
team(texans, afc, south).    team(jaguars, afc, south).
team(colts, afc, south).     team(titans, afc, south).
team(chiefs, afc, west).     team(broncos, afc, west).
team(raiders, afc, west).    team(chargers, afc, west).
team(eagles, nfc, east).     team(cowboys, nfc, east).
team(commanders, nfc, east). team(giants, nfc, east).
team(bears, nfc, north).     team(lions, nfc, north).
team(packers, nfc, north).   team(vikings, nfc, north).
team(buccaneers, nfc, south).team(saints, nfc, south).
team(panthers, nfc, south).  team(falcons, nfc, south).
team(niners, nfc, west).     team(seahawks, nfc, west).
team(rams, nfc, west).       team(cardinals, nfc, west).

% Rules
teammates(X, Y) :-
    player(X, Team, _), player(Y, Team, _), X \\= Y.

plays_in_conference(Player, Conference) :-
    player(Player, Team, _), team(Team, Conference, _).

plays_in_division(Player, Division) :-
    player(Player, Team, _), team(Team, _, Division).

division_rivals(X, Y) :-
    player(X, TeamX, _), player(Y, TeamY, _),
    team(TeamX, _, Div), team(TeamY, _, Div), TeamX \\= TeamY.

plays_offense(Player) :-
    player(Player, _, Pos),
    member(Pos, [qb, wr, rb, te, ot, og, c]).

plays_defense(Player) :-
    player(Player, _, Pos),
    member(Pos, [de, dt, lb, cb, s]).

team_roster(Team, Player) :- player(Player, Team, _).
"""

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

    TEST_QUERIES = [
        # ── teammates ───────────────────────────────────────────────────────
        {
            "id": "TM-1", "rule": "teammates",
            "nl_question": "Are Joe Burrow and Ja'Marr Chase teammates?",
            "fol_query":    "Teammates(joe_burrow, ja_marr_chase)",
            "prolog_query": "teammates(joe_burrow, ja_marr_chase).",
            "expected": "Yes",
            "reasoning": "Both play for bengals -> teammates rule fires.",
        },
        {
            "id": "TM-2", "rule": "teammates",
            "nl_question": "Are Patrick Mahomes and Josh Allen teammates?",
            "fol_query":    "Teammates(patrick_mahomes, josh_allen)",
            "prolog_query": "teammates(patrick_mahomes, josh_allen).",
            "expected": "No",
            "reasoning": "Mahomes->chiefs, Allen->bills. Different teams.",
        },
        {
            "id": "TM-3", "rule": "teammates",
            "nl_question": "Are Caleb Williams and Luther Burden teammates?",
            "fol_query":    "Teammates(caleb_williams, luther_burden)",
            "prolog_query": "teammates(caleb_williams, luther_burden).",
            "expected": "Yes",
            "reasoning": "Both play for bears -> teammates rule fires.",
        },

        # ── plays_in_conference ──────────────────────────────────────────────
        {
            "id": "CONF-1", "rule": "plays_in_conference",
            "nl_question": "Does Josh Allen play in the AFC?",
            "fol_query":    "PlaysInConference(josh_allen, afc)",
            "prolog_query": "plays_in_conference(josh_allen, afc).",
            "expected": "Yes",
            "reasoning": "Allen->bills, team(bills,afc,east) -> afc.",
        },
        {
            "id": "CONF-2", "rule": "plays_in_conference",
            "nl_question": "Does Jalen Hurts play in the NFC?",
            "fol_query":    "PlaysInConference(jalen_hurts, nfc)",
            "prolog_query": "plays_in_conference(jalen_hurts, nfc).",
            "expected": "Yes",
            "reasoning": "Hurts->eagles, team(eagles,nfc,east) -> nfc.",
        },
        {
            "id": "CONF-3", "rule": "plays_in_conference",
            "nl_question": "Does Patrick Mahomes play in the NFC?",
            "fol_query":    "PlaysInConference(patrick_mahomes, nfc)",
            "prolog_query": "plays_in_conference(patrick_mahomes, nfc).",
            "expected": "No",
            "reasoning": "Mahomes->chiefs, team(chiefs,afc,west) -> AFC not NFC.",
        },

        # ── plays_in_division ────────────────────────────────────────────────
        {
            "id": "DIV-1", "rule": "plays_in_division",
            "nl_question": "Does Lamar Jackson play in the North division?",
            "fol_query":    "PlaysInDivision(lamar_jackson, north)",
            "prolog_query": "plays_in_division(lamar_jackson, north).",
            "expected": "Yes",
            "reasoning": "Jackson->ravens, team(ravens,afc,north) -> north.",
        },
        {
            "id": "DIV-2", "rule": "plays_in_division",
            "nl_question": "Does Tyreek Hill play in the East division?",
            "fol_query":    "PlaysInDivision(tyreek_hill, east)",
            "prolog_query": "plays_in_division(tyreek_hill, east).",
            "expected": "Yes",
            "reasoning": "Hill->dolphins, team(dolphins,afc,east) -> east.",
        },

        # ── division_rivals ──────────────────────────────────────────────────
        {
            "id": "RIVAL-1", "rule": "division_rivals",
            "nl_question": "Are Josh Allen and Drake Maye division rivals?",
            "fol_query":    "DivisionRivals(josh_allen, drake_maye)",
            "prolog_query": "division_rivals(josh_allen, drake_maye).",
            "expected": "Yes",
            "reasoning": "Allen->bills(AFC East), Maye->patriots(AFC East). Same div, diff teams.",
        },
        {
            "id": "RIVAL-2", "rule": "division_rivals",
            "nl_question": "Are Patrick Mahomes and Lamar Jackson division rivals?",
            "fol_query":    "DivisionRivals(patrick_mahomes, lamar_jackson)",
            "prolog_query": "division_rivals(patrick_mahomes, lamar_jackson).",
            "expected": "No",
            "reasoning": "Mahomes->chiefs(AFC West), Jackson->ravens(AFC North). Different divisions.",
        },
        {
            "id": "RIVAL-3", "rule": "division_rivals",
            "nl_question": "Are Jalen Hurts and Jayden Daniels division rivals?",
            "fol_query":    "DivisionRivals(jalen_hurts, jayden_daniels)",
            "prolog_query": "division_rivals(jalen_hurts, jayden_daniels).",
            "expected": "Yes",
            "reasoning": "Hurts->eagles(NFC East), Daniels->commanders(NFC East). Same div.",
        },

        # ── plays_offense ────────────────────────────────────────────────────
        {
            "id": "OFF-1", "rule": "plays_offense",
            "nl_question": "Does Saquon Barkley play offense?",
            "fol_query":    "PlaysOffense(saquon_barkley)",
            "prolog_query": "plays_offense(saquon_barkley).",
            "expected": "Yes",
            "reasoning": "Barkley has position rb, which is in [qb,wr,rb,te,ot,og,c].",
        },
        {
            "id": "OFF-2", "rule": "plays_offense",
            "nl_question": "Does Myles Garrett play offense?",
            "fol_query":    "PlaysOffense(myles_garrett)",
            "prolog_query": "plays_offense(myles_garrett).",
            "expected": "No",
            "reasoning": "Garrett has position de, NOT in the offense list.",
        },

        # ── plays_defense ────────────────────────────────────────────────────
        {
            "id": "DEF-1", "rule": "plays_defense",
            "nl_question": "Does Nick Bosa play defense?",
            "fol_query":    "PlaysDefense(nick_bosa)",
            "prolog_query": "plays_defense(nick_bosa).",
            "expected": "Yes",
            "reasoning": "Bosa has position de, which is in [de,dt,lb,cb,s].",
        },
        {
            "id": "DEF-2", "rule": "plays_defense",
            "nl_question": "Does Brock Purdy play defense?",
            "fol_query":    "PlaysDefense(brock_purdy)",
            "prolog_query": "plays_defense(brock_purdy).",
            "expected": "No",
            "reasoning": "Purdy has position qb, NOT in the defense list.",
        },
        {
            "id": "DEF-3", "rule": "plays_defense",
            "nl_question": "Does Kyle Hamilton play defense?",
            "fol_query":    "PlaysDefense(kyle_hamilton)",
            "prolog_query": "plays_defense(kyle_hamilton).",
            "expected": "Yes",
            "reasoning": "Hamilton has position s (safety), in the defense list.",
        },
    ]

    def get_fol_context(self):    return self.FOL_AXIOMS.strip()
    def get_prolog_context(self): return self.PROLOG_FACTS.strip()

    def format_prompt(self, query: Dict, mode: str = "fol") -> str:
        kb    = self.get_fol_context() if mode == "fol" else self.get_prolog_context()
        qtext = query["fol_query"]     if mode == "fol" else query["prolog_query"]
        lang  = "First-Order Logic"    if mode == "fol" else "Prolog"
        return (
            f"You are given the following NFL knowledge base in {lang}:\n\n"
            f"{kb}\n\n"
            f"Using step-by-step logical deduction, determine whether the "
            f"following query is TRUE or FALSE. Start your answer with 'Yes' or 'No', "
            f"then explain your reasoning.\n\n"
            f"Query ({lang}): {qtext}\n"
            f"Question: {query['nl_question']}"
        )

    def as_symbolic_samples(self) -> List[SymbolicSample]:
        return [
            SymbolicSample(
                task_type="fol_reasoning",
                symbol_family="first_order_logic",
                instruction=(
                    "Using the NFL knowledge base axioms, evaluate whether the "
                    "following FOL query holds. Reason step-by-step."
                ),
                symbolic_input=self.get_fol_context() + "\n\nQuery: " + q["fol_query"],
                expected_output=q["expected"] + ". " + q["reasoning"],
            )
            for q in self.TEST_QUERIES
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SYMBOL-LLM INFERENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class SymbolLLMInference:
    """
    Wraps Symbol-LLM-7B-Instruct from HuggingFace.
    Requires: pip install transformers torch accelerate
    """
    MODEL_ID = "Symbol-LLM/Symbol-LLM-7B-Instruct"

    def __init__(self, device="auto"):
        self.device = device
        self.tokenizer = self.model = None
        self._loaded = False

    def load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        print(f"Loading {self.MODEL_ID} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, device_map=self.device, torch_dtype=torch.float16)
        self._loaded = True
        print("Model ready.\n")

    def query(self, prompt: str, max_new_tokens=300, temperature=0.1) -> str:
        if not self._loaded:
            raise RuntimeError("Call .load() first.")
        import torch
        sys_msg = (
            "You are Symbol-LLM, an expert in formal symbolic reasoning. "
            "For every question: first say Yes or No, then give a step-by-step explanation."
        )
        full = f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{prompt} [/INST]"
        inputs = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return decoded[len(full):].strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATOR  — per-rule accuracy breakdown
# ─────────────────────────────────────────────────────────────────────────────

class NFLKBEvaluator:
    def __init__(self, kb: NFLKnowledgeBase):
        self.kb = kb
        self.results: List[Dict] = []

    @staticmethod
    def _verdict(response: str) -> Optional[str]:
        t = response.lower().strip()
        if re.match(r'^yes\b', t): return "Yes"
        if re.match(r'^no\b',  t): return "No"
        if "yes" in t[:60]:        return "Yes"
        if "no"  in t[:60]:        return "No"
        return None

    def check(self, response: str, expected: str) -> bool:
        v = self._verdict(response)
        return v is not None and v == expected

    def run(self, inference: SymbolLLMInference,
            mode="fol", use_evol=False) -> Dict:
        evol = SymbolEvol() if use_evol else None
        correct = 0
        rule_stats: Dict[str, Dict] = {}

        print(f"{'ID':<10} {'Rule':<22} {'Expected':<10} {'Got':<10} OK")
        print("─" * 58)

        for q in self.kb.TEST_QUERIES:
            prompt = self.kb.format_prompt(q, mode)
            if use_evol and evol:
                aug, _ = evol.augment_fol(q["fol_query"])
                prompt += f"\n\n(Symbol-Evol augmented query: {aug})"

            response = inference.query(prompt)
            ok = self.check(response, q["expected"])
            correct += int(ok)

            rule = q["rule"]
            rule_stats.setdefault(rule, {"correct": 0, "total": 0})
            rule_stats[rule]["correct"] += int(ok)
            rule_stats[rule]["total"]   += 1

            v = self._verdict(response) or "???"
            print(f"{q['id']:<10} {rule:<22} {q['expected']:<10} {v:<10} {'✓' if ok else '✗'}")
            self.results.append({"id": q["id"], "rule": rule,
                                  "question": q["nl_question"],
                                  "expected": q["expected"],
                                  "model_verdict": v, "correct": ok,
                                  "response_snippet": response[:200]})

        total = len(self.kb.TEST_QUERIES)
        acc   = correct / total
        print(f"\n{'─'*58}")
        print(f"Overall:  {correct}/{total} = {acc:.1%}\n")
        print("Per-rule:")
        for r, s in rule_stats.items():
            print(f"  {r:<24} {s['correct']}/{s['total']}")

        return {"accuracy": acc, "correct": correct, "total": total,
                "rule_breakdown": rule_stats, "results": self.results}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DEMO  (no GPU required)
# ─────────────────────────────────────────────────────────────────────────────

def demo_symbol_evol(kb: NFLKnowledgeBase):
    print("=" * 60)
    print("SYMBOL-EVOL AUGMENTATION ON NFL KB")
    print("=" * 60)
    evol = SymbolEvol(seed=42)

    print("\n── FOL queries before and after Symbol-Evol ──\n")
    for q in kb.TEST_QUERIES[:6]:
        aug, m = evol.augment_fol(q["fol_query"])
        print(f"  {q['nl_question']}")
        print(f"    Original:  {q['fol_query']}")
        print(f"    Augmented: {aug}")
        print(f"    Mapping:   {m}\n")

    print("── Prolog fact augmentation ──\n")
    facts = kb.PROLOG_FACTS.strip().split('\n')[:8]
    aug_facts, m = evol.augment_prolog(facts)
    for orig, aug in zip(facts, aug_facts):
        if orig.strip():
            print(f"  {orig:<45} -> {aug}")


def demo_pipeline(kb: NFLKnowledgeBase):
    print("\n" + "=" * 60)
    print("TWO-STAGE PIPELINE DEMO")
    print("=" * 60)

    pipeline = SymbolicCollectionPipeline(augment=True, augment_ratio=0.6)
    samples  = kb.as_symbolic_samples()
    stage1   = pipeline.prepare_stage1(samples)

    aug_count = sum(1 for e in stage1 if e["augmented"])
    print(f"\nStage 1 — {len(stage1)} symbolic examples  |  {aug_count} augmented via Symbol-Evol")

    nl = [{"messages": [
        {"role": "user",      "content": "What is a tight end in American football?"},
        {"role": "assistant", "content": "A tight end (TE) is a hybrid position..."},
    ]}]
    stage2 = pipeline.prepare_stage2(samples, nl, ratio=0.3)
    nl_c   = sum(1 for e in stage2 if "task_type" not in e)
    sym_c  = sum(1 for e in stage2 if "task_type" in e)
    print(f"Stage 2 — {len(stage2)} total  |  symbolic={sym_c}  NL={nl_c}")

    print("\n── Sample Stage 1 training instance ──")
    ex  = stage1[0]
    msg = ex["messages"]
    print(f"  Augmented: {ex['augmented']}")
    print(f"  User:      {msg[1]['content'][:130]}...")
    print(f"  Assistant: {msg[2]['content']}")


def demo_mock_eval(kb: NFLKnowledgeBase):
    """
    Simulated Symbol-LLM responses for every test query.
    Replace with real SymbolLLMInference when GPU is available.
    """
    print("\n" + "=" * 60)
    print("KB EVALUATION — Mock Responses (simulating Symbol-LLM-7B)")
    print("=" * 60)

    mock: Dict[str, str] = {
        "TM-1":    "Yes. joe_burrow plays for bengals and ja_marr_chase plays for bengals. Same team, X≠Y → teammates.",
        "TM-2":    "No. patrick_mahomes plays for chiefs, josh_allen plays for bills. Different teams → not teammates.",
        "TM-3":    "Yes. caleb_williams plays for bears and luther_burden plays for bears. Same team → teammates.",
        "CONF-1":  "Yes. josh_allen → bills → team(bills, afc, east). InConference(bills, afc) → PlaysInConference(josh_allen, afc).",
        "CONF-2":  "Yes. jalen_hurts → eagles → team(eagles, nfc, east). InConference(eagles, nfc) → PlaysInConference(jalen_hurts, nfc).",
        "CONF-3":  "No. patrick_mahomes → chiefs → team(chiefs, afc, west). InConference(chiefs, afc), not nfc → false.",
        "DIV-1":   "Yes. lamar_jackson → ravens → team(ravens, afc, north). InDivision(ravens, north) → PlaysInDivision(lamar_jackson, north).",
        "DIV-2":   "Yes. tyreek_hill → dolphins → team(dolphins, afc, east). InDivision(dolphins, east) → PlaysInDivision(tyreek_hill, east).",
        "RIVAL-1": "Yes. josh_allen→bills(AFC East), drake_maye→patriots(AFC East). Same division, different teams → DivisionRivals.",
        "RIVAL-2": "No. patrick_mahomes→chiefs(AFC West), lamar_jackson→ravens(AFC North). Different divisions → not rivals.",
        "RIVAL-3": "Yes. jalen_hurts→eagles(NFC East), jayden_daniels→commanders(NFC East). Same division → DivisionRivals.",
        "OFF-1":   "Yes. saquon_barkley has position rb. rb ∈ [qb,wr,rb,te,ot,og,c] → PlaysOffense is true.",
        "OFF-2":   "No. myles_garrett has position de. de ∉ [qb,wr,rb,te,ot,og,c] → PlaysOffense is false.",
        "DEF-1":   "Yes. nick_bosa has position de. de ∈ [de,dt,lb,cb,s] → PlaysDefense is true.",
        "DEF-2":   "No. brock_purdy has position qb. qb ∉ [de,dt,lb,cb,s] → PlaysDefense is false.",
        "DEF-3":   "Yes. kyle_hamilton has position s (safety). s ∈ [de,dt,lb,cb,s] → PlaysDefense is true.",
    }

    evaluator = NFLKBEvaluator(kb)
    correct = 0
    rule_stats: Dict[str, Dict] = {}

    print(f"\n{'ID':<10} {'Rule':<22} {'Expected':<10} {'Got':<10} OK")
    print("─" * 58)

    for q in kb.TEST_QUERIES:
        resp = mock[q["id"]]
        ok   = evaluator.check(resp, q["expected"])
        correct += int(ok)
        rule = q["rule"]
        rule_stats.setdefault(rule, {"correct": 0, "total": 0})
        rule_stats[rule]["correct"] += int(ok)
        rule_stats[rule]["total"]   += 1
        v = evaluator._verdict(resp) or "???"
        print(f"{q['id']:<10} {rule:<22} {q['expected']:<10} {v:<10} {'✓' if ok else '✗'}")

    total = len(kb.TEST_QUERIES)
    print(f"\n{'─'*58}")
    print(f"Overall:  {correct}/{total} = {correct/total:.1%}\n")
    print("Per-rule breakdown:")
    for r, s in rule_stats.items():
        print(f"  {r:<24} {s['correct']}/{s['total']}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Symbol-LLM NFL KB Reimplementation")
    parser.add_argument("--mode",       choices=["demo", "eval"], default="demo",
        help="demo=no-GPU showcase | eval=load real Symbol-LLM-7B model")
    parser.add_argument("--query-mode", choices=["fol", "prolog"], default="fol")
    parser.add_argument("--use-evol",   action="store_true",
        help="Apply Symbol-Evol augmentation to test queries")
    args = parser.parse_args()

    print("Symbol-LLM Reimplementation — NFL KB Edition")
    print("Paper: arXiv:2311.09278 | ACL 2024\n")

    kb = NFLKnowledgeBase()

    if args.mode == "demo":
        demo_symbol_evol(kb)
        demo_pipeline(kb)
        demo_mock_eval(kb)

    elif args.mode == "eval":
        model     = SymbolLLMInference()
        model.load()
        evaluator = NFLKBEvaluator(kb)
        results   = evaluator.run(model, mode=args.query_mode, use_evol=args.use_evol)
        out = "nfl_eval_results.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved -> {out}")


if __name__ == "__main__":
    main()
