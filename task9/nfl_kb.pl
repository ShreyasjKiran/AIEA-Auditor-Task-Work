% NFL Knowledge Base
% Facts are in the form: player(Name, Team, Position).

% Quarterbacks
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

% Wide Receivers
player(ja_marr_chase, bengals, wr).
player(ceedee_lamb, cowboys, wr).
player(tyreek_hill, dolphins, wr).
player(puka_nacua, rams, wr).
player(malik_nabers, giants, wr).
player(jaxon_smith_njigba, seahawks, wr).
player(george_pickens, cowboys, wr).
player(luther_burden, bears, wr).

% Running Backs
player(saquon_barkley, eagles, rb).
player(derrick_henry, ravens, rb).
player(breece_hall, jets, rb).
player(travis_etienne, jaguars, rb).
player(kenneth_walker, seahawks, rb).

% Tight Ends
player(brock_bowers, raiders, te).
player(trey_mcbride, cardinals, te).

% Defensive Players
player(myles_garrett, browns, de).
player(trey_hendrickson, bengals, de).
player(josh_hines_allen, jaguars, de).
player(maxx_crosby, raiders, de).
player(jared_verse, rams, de).
player(nick_bosa, niners, de).
player(patrick_surtain, broncos, cb).
player(devon_witherspoon, seahawks, cb).
player(kyle_hamilton, ravens, s).

% Offensive Linemen
player(dion_dawkins, bills, ot).

% --- Team Info ---
% team(Name, Conference, Division).
team(bills, afc, east).
team(dolphins, afc, east).
team(patriots, afc, east).
team(jets, afc, east).
team(ravens, afc, north).
team(bengals, afc, north).
team(browns, afc, north).
team(steelers, afc, north).
team(texans, afc, south).
team(jaguars, afc, south).
team(colts, afc, south).
team(titans, afc, south).
team(chiefs, afc, west).
team(broncos, afc, west).
team(raiders, afc, west).
team(chargers, afc, west).
team(eagles, nfc, east).
team(cowboys, nfc, east).
team(commanders, nfc, east).
team(giants, nfc, east).
team(bears, nfc, north).
team(lions, nfc, north).
team(packers, nfc, north).
team(vikings, nfc, north).
team(buccaneers, nfc, south).
team(saints, nfc, south).
team(panthers, nfc, south).
team(falcons, nfc, south).
team(niners, nfc, west).
team(seahawks, nfc, west).
team(rams, nfc, west).
team(cardinals, nfc, west).

% --- Rules ---

% Two players are teammates if they play for the same team.
teammates(X, Y) :-
    player(X, Team, _),
    player(Y, Team, _),
    X \= Y.

% A player plays in a given conference.
plays_in_conference(Player, Conference) :-
    player(Player, Team, _),
    team(Team, Conference, _).

% A player plays in a given division.
plays_in_division(Player, Division) :-
    player(Player, Team, _),
    team(Team, _, Division).

% Two players are division rivals if they are in the same division but different teams.
division_rivals(X, Y) :-
    player(X, TeamX, _),
    player(Y, TeamY, _),
    team(TeamX, _, Div),
    team(TeamY, _, Div),
    TeamX \= TeamY.

% Check if a player plays offense.
plays_offense(Player) :-
    player(Player, _, Pos),
    member(Pos, [qb, wr, rb, te, ot, og, c]).

% Check if a player plays defense.
plays_defense(Player) :-
    player(Player, _, Pos),
    member(Pos, [de, dt, lb, cb, s]).

% Find all players on a given team.
team_roster(Team, Player) :-
    player(Player, Team, _).
