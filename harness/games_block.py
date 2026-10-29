# ---- synthetic game tasks (generated, exactly scored, no outside data): games:nim, games:subtraction, games:sokoban, games:kuhn
def _nim_gen(rng, heaps=(4, 7), mx=40):
    k = rng.randint(*heaps); h = [rng.randint(1, mx) for _ in range(k)]
    x = 0
    for v in h: x ^= v
    q = (f"Nim (normal play: the player who takes the last stone wins). Heaps: {h}. It is your move; a move removes 1 or more stones from exactly one heap. "
         f"If a winning move exists, give one as heap_index:stones_removed with 1-based heap index (for example \\boxed{{2:5}}). If no winning move exists, answer \\boxed{{none}}.")
    return q, dict(kind="game", game="nim", heaps=h, pposition=(x == 0))
def _nim_score(ans, g):
    h = g["heaps"]
    if g["pposition"]: return float(ans.strip().lower() == "none")
    try: i, r = [int(t) for t in ans.replace(" ", "").split(":")]
    except Exception: return 0.0
    if not (1 <= i <= len(h) and 1 <= r <= h[i - 1]): return 0.0
    x = 0
    for j, v in enumerate(h): x ^= (v - r if j == i - 1 else v)
    return float(x == 0)
def _sub_gen(rng, nmax=120):
    S = sorted(rng.sample(range(1, 12), rng.randint(3, 4))); n = rng.randint(20, nmax)
    P = [True] + [False] * n  # P[m]: position with m stones is a P-position (previous player wins) under normal play
    for m in range(1, n + 1): P[m] = not any(m - s >= 0 and P[m - s] for s in S)
    q = (f"Subtraction game (normal play: the player who cannot move loses). A single pile has {n} stones; on each turn a player removes exactly one of {S} stones (the move must not exceed the pile). "
         f"It is your move. If a winning move exists, give the number of stones to remove (for example \\boxed{{{S[0]}}}). If no winning move exists, answer \\boxed{{none}}.")
    return q, dict(kind="game", game="subtraction", n=n, S=S, P=P)
def _sub_score(ans, g):
    n, S, P = g["n"], g["S"], g["P"]
    if P[n]: return float(ans.strip().lower() == "none")
    try: s = int(ans.strip())
    except Exception: return 0.0
    return float(s in S and s <= n and P[n - s])
def _sok_gen(rng, size=6, min_len=6, max_tries=200):
    from collections import deque
    D = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    for _ in range(max_tries):
        walls = {(r, c) for r in range(size) for c in range(size) if r in (0, size - 1) or c in (0, size - 1)}
        for _ in range(rng.randint(2, 5)): walls.add((rng.randint(1, size - 2), rng.randint(1, size - 2)))
        free = [p for p in [(r, c) for r in range(size) for c in range(size)] if p not in walls]
        if len(free) < 6: continue
        goal, box, player = rng.sample(free, 3)
        def solve():
            start = (player, box); seen = {start}; dq = deque([(start, "")])
            while dq:
                (pl, bx), path = dq.popleft()
                if bx == goal: return path
                if len(path) > 40: continue
                for m, (dr, dc) in D.items():
                    np_ = (pl[0] + dr, pl[1] + dc)
                    if np_ in walls: continue
                    nb = bx
                    if np_ == bx:
                        nb = (bx[0] + dr, bx[1] + dc)
                        if nb in walls: continue
                    st = (np_, nb)
                    if st not in seen: seen.add(st); dq.append((st, path + m))
            return None
        sol = solve()
        if sol is None or len(sol) < min_len: continue
        grid = [["#" if (r, c) in walls else "." for c in range(size)] for r in range(size)]
        grid[goal[0]][goal[1]] = "G"; grid[box[0]][box[1]] = "B"; grid[player[0]][player[1]] = "P"
        board = "\n".join("".join(row) for row in grid)
        q = (f"Sokoban. Grid (# wall, . floor, P you, B box, G goal):\n{board}\nMoves are U, D, L, R (up, down, left, right). You push the box by walking into it; the box moves one cell in the same direction if that cell is not a wall. "
             f"You cannot walk into walls. Give a sequence of moves that ends with the box on the goal, as a string of letters inside \\boxed{{}} (for example \\boxed{{RRDL}}). Shortest solution has {len(sol)} moves; any valid solution of at most {len(sol) + 6} moves is accepted.")
        return q, dict(kind="game", game="sokoban", walls=sorted(walls), goal=goal, box=box, player=player, maxlen=len(sol) + 6)
    raise RuntimeError("sokoban gen failed")
def _sok_score(ans, g):
    D = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    walls = {tuple(w) for w in g["walls"]}; pl = tuple(g["player"]); bx = tuple(g["box"]); goal = tuple(g["goal"])
    s = ans.strip().upper().replace(" ", "").replace(",", "")
    if not s or len(s) > g["maxlen"] or any(ch not in D for ch in s): return 0.0
    for ch in s:
        dr, dc = D[ch]; np_ = (pl[0] + dr, pl[1] + dc)
        if np_ in walls: return 0.0
        if np_ == bx:
            nb = (bx[0] + dr, bx[1] + dc)
            if nb in walls: return 0.0
            bx = nb
        pl = np_
    return float(bx == goal)
# Kuhn poker: exact-EV scoring vs the standard Nash opponent (alpha = 1/3). Cards J<Q<K, ante 1, bet 1. P1 acts first.
def _kuhn_nash(alpha=1.0 / 3):
    # P1: bet probability on first action by card; after check-bet: call probability by card
    p1_bet = {"J": alpha, "Q": 0.0, "K": 3 * alpha}; p1_call = {"J": 0.0, "Q": alpha + 1.0 / 3, "K": 1.0}
    # P2 facing check: bet prob by card; facing bet: call prob by card
    p2_bet = {"J": 1.0 / 3, "Q": 0.0, "K": 1.0}; p2_call = {"J": 0.0, "Q": 1.0 / 3, "K": 1.0}
    return p1_bet, p1_call, p2_bet, p2_call
def _kuhn_ev(card, pos, history, action):
    """EV (in chips, from the acting player's view) of `action` at (card, pos, history) vs the Nash opponent, averaging over the opponent's card."""
    R = {"J": 0, "Q": 1, "K": 2}; p1_bet, p1_call, p2_bet, p2_call = _kuhn_nash()
    def showdown(me, opp, pot_win): return pot_win if R[me] > R[opp] else -pot_win
    ev = 0.0; opps = [c for c in "JQK" if c != card]
    for opp in opps:
        if pos == 1 and history == "":
            if action == "check":
                pb = p2_bet[opp]
                # opp checks -> showdown for 1; opp bets -> I respond per my Nash call prob? No: EV of *this* action assumes I later play Nash.
                after_bet = p1_call[card] * showdown(card, opp, 2) + (1 - p1_call[card]) * (-1)
                ev += (1 - pb) * showdown(card, opp, 1) + pb * after_bet
            else:  # bet
                pc = p2_call[opp]; ev += pc * showdown(card, opp, 2) + (1 - pc) * 1
        elif pos == 2 and history == "check":
            if action == "check": ev += showdown(card, opp, 1)
            else:
                pc = p1_call[opp]; ev += pc * showdown(card, opp, 2) + (1 - pc) * 1
        elif pos == 2 and history == "bet":
            ev += showdown(card, opp, 2) if action == "call" else -1
        elif pos == 1 and history == "check,bet":
            ev += showdown(card, opp, 2) if action == "call" else -1
        else: raise ValueError(history)
    return ev / len(opps)
def _kuhn_gen(rng):
    card = rng.choice("JQK"); spot = rng.choice([(1, ""), (2, "check"), (2, "bet"), (1, "check,bet")])
    pos, hist = spot; acts = ["check", "bet"] if hist in ("", "check") else ["call", "fold"]
    evs = {a: _kuhn_ev(card, pos, hist, a) for a in acts}; best = max(evs.values()); good = [a for a in acts if evs[a] >= best - 1e-9]
    hdesc = {"": "You act first; no actions yet.", "check": "Player 1 checked.", "bet": "Player 1 bet 1.", "check,bet": "You checked, then Player 2 bet 1."}[hist]
    q = (f"Kuhn poker: 3-card deck (J < Q < K), each player antes 1, one card each, one betting round with a fixed bet of 1; higher card wins at showdown. "
         f"Your opponent plays the standard Nash equilibrium strategy (bluffs a J by betting 1/3 of the time, always bets a K, calls a bet with Q 1/3 of the time, always calls with K, never calls with J). "
         f"You are Player {pos} holding {card}. {hdesc} Your options: {' or '.join(acts)}. Which action maximizes your expected chips against this opponent? Answer with one word inside \\boxed{{}}.")
    return q, dict(kind="game", game="kuhn", good=good, evs=evs)
def _kuhn_score(ans, g): return float(ans.strip().lower().rstrip(".") in g["good"])
GAMES = {"nim": (_nim_gen, _nim_score), "subtraction": (_sub_gen, _sub_score), "sokoban": (_sok_gen, _sok_score), "kuhn": (_kuhn_gen, _kuhn_score)}
def _reward_game(text, gold):
    pred = last_boxed(text)
    if pred is None: return 0.0
    try: return float(GAMES[gold["game"]][1](pred, gold))
    except Exception: return 0.0
def build_game_prompts(fmt, name, n_train, n_eval, seed):
    import random
    gen = GAMES[name][0]; names = list(GAMES) if name == "mix" else None
    def make(rng, n):
        out = []
        for i in range(n):
            g = GAMES[names[i % len(names)]][0] if names else gen
            q, gold = g(rng); out.append((fmt(q), gold))
        return out
    return make(random.Random(1000 + seed), n_train), make(random.Random(7), n_eval)
