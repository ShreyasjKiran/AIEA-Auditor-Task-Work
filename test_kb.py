import janus_swi as janus

# Load the knowledge base
janus.consult("nfl_kb.pl")

# Simple true/false query
result = janus.query_once("player(josh_allen, bills, qb)")
print("Josh Allen is Bills QB:", result["truth"])

# Query with variables - find all Seahawks players
print("\nSeahawks players:")
for result in janus.query("player(X, seahawks, _)"):
    print(f"  {result['X']}")

# Test a rule - teammates
print("\nJa'Marr Chase's teammates:")
for result in janus.query("teammates(ja_marr_chase, X)"):
    print(f"  {result['X']}")

# Test division rivals
result = janus.query_once("division_rivals(joe_burrow, lamar_jackson)")
print(f"\nBurrow and Jackson are division rivals: {result['truth']}")

# Test plays_defense
print("\nDefensive players:")
for result in janus.query("plays_defense(X)"):
    print(f"  {result['X']}")
