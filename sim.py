"""
Accurate trade simulation from logs.
Replicates all bot logic: trailing SL, consecutive SL pause,
dynamic threshold, per-pair SL cooldown, EARLY_EXIT, etc.
"""

import re
import json
import sys
from datetime import datetime
from collections import defaultdict


# ── Parse logs ────────────────────────────────────────────────

def parse_logs(log_path):
    scans, refineds, refined_invs = [], [], []
    candles_1m = defaultdict(list)  # sym → list of {ts, high, low, close}

    with open(log_path) as f:
        for line in f:
            ts_m = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if not ts_m:
                continue
            ts = datetime.strptime(ts_m.group(1), '%Y-%m-%d %H:%M:%S')

            if '| CANDLE_1M ' in line:
                m = re.search(r'CANDLE_1M (\S+)', line)
                c_ts = re.search(r'ts=(\S+ \S+)', line)
                c_h = re.search(r'high=([0-9.]+)', line)
                c_l = re.search(r'low=([0-9.]+)', line)
                c_c = re.search(r'close=([0-9.]+)', line)
                if m and c_h and c_l and c_c:
                    candle_ts = datetime.strptime(c_ts.group(1), '%Y-%m-%d %H:%M') if c_ts else ts
                    candles_1m[m.group(1)].append({
                        'ts': candle_ts,
                        'high': float(c_h.group(1)),
                        'low': float(c_l.group(1)),
                        'close': float(c_c.group(1)),
                    })
                continue

            if '| SCAN ' in line:
                m = re.search(r'SCAN (\S+)', line)
                if not m:
                    continue
                p = re.search(r'price=([0-9.]+)', line)
                h = re.search(r'high=([0-9.]+)', line)
                l = re.search(r'low=([0-9.]+)', line)
                a = re.search(r'\| atr=([0-9.]+)', line)
                rg = re.search(r'regime=(\S+)', line)
                ml_m = re.search(r'ml=(BUY|SELL|HOLD)\(([0-9.]+)% disagr=(\d+)%\)', line)
                if not all([p, h, l, a]):
                    continue
                scan = {
                    'ts': ts, 'sym': m.group(1),
                    'price': float(p.group(1)), 'high': float(h.group(1)),
                    'low': float(l.group(1)), 'atr': float(a.group(1)),
                    'regime': rg.group(1) if rg else '?',
                }
                if ml_m:
                    scan['ml_dir'] = ml_m.group(1)
                    scan['ml_conf'] = float(ml_m.group(2)) / 100
                    scan['ml_disagr'] = int(ml_m.group(3)) / 100
                scans.append(scan)

            elif '| REFINED_INV ' in line:
                r = _parse_refined(line, ts, 'REFINED_INV')
                if r:
                    refined_invs.append(r)

            elif '| REFINED ' in line:
                r = _parse_refined(line, ts, 'REFINED')
                if r:
                    refineds.append(r)

    # Deduplicate and sort 1m candles
    for sym in candles_1m:
        seen = set()
        unique = []
        for c in candles_1m[sym]:
            key = c['ts']
            if key not in seen:
                seen.add(key)
                unique.append(c)
        candles_1m[sym] = sorted(unique, key=lambda x: x['ts'])

    return scans, refineds, refined_invs, candles_1m


def _parse_refined(line, ts, tag):
    m = re.search(rf'{tag} (\S+)', line)
    if not m:
        return None
    e = re.search(r'entry=([0-9.]+)', line)
    a = re.search(r'atr=([0-9.]+)', line)
    c = re.search(r'conf=([0-9.]+)%', line)
    d = re.search(r'disagr=(\d+)%', line)
    dr = re.search(r'dir=(\S+)', line)
    rg = re.search(r'regime=(\S+)', line)
    if not all([e, a, c, dr]):
        return None
    return {
        'ts': ts, 'sym': m.group(1),
        'entry': float(e.group(1)), 'atr': float(a.group(1)),
        'conf': float(c.group(1)) / 100,
        'disagr': int(d.group(1)) / 100 if d else 0,
        'dir': dr.group(1),
        'regime': rg.group(1) if rg else '?',
    }


# ── Build data structures ────────────────────────────────────

def build_structures(scans, refineds, refined_invs, candles_1m=None):
    sym_scans = defaultdict(list)
    for s in scans:
        sym_scans[s['sym']].append(s)
    for sym in sym_scans:
        sym_scans[sym].sort(key=lambda x: x['ts'])

    rounds = []
    cur = []
    for s in sorted(scans, key=lambda x: x['ts']):
        if cur and (s['ts'] - cur[0]['ts']).total_seconds() > 30:
            rounds.append(cur)
            cur = []
        cur.append(s)
    if cur:
        rounds.append(cur)

    def map_to_rounds(items):
        by_round = defaultdict(list)
        for r in items:
            idx = min(range(len(rounds)),
                      key=lambda i: abs((rounds[i][0]['ts'] - r['ts']).total_seconds()))
            by_round[idx].append(r)
        return by_round

    return sym_scans, rounds, map_to_rounds(refineds), map_to_rounds(refined_invs)


# ── Simulation ────────────────────────────────────────────────

def simulate(rounds, sym_scans, ref_by_round, config, candles_1m=None, verbose=False):
    threshold_base = config.get('threshold', 0.80)
    max_open = config.get('max_open', 2)
    cooldown = config.get('cooldown', 4)
    timeout = config.get('timeout', 18)
    dead_hours = config.get('dead_hours', list(range(13, 16)))
    regime_params = config.get('regime_params', {
        'TREND': {'sl': 1.5, 'tp': 3.0},
        'RANGE': {'sl': 1.5, 'tp': 3.0},
        'REVERSAL': {'sl': 1.5, 'tp': 3.0},
    })
    trail_activation = config.get('trail_activation', 1.5)
    trail_distance = config.get('trail_distance', 1.0)
    trail_enabled = config.get('trail_enabled', True)
    skip_regimes = config.get('skip_regimes', [])

    # Bot state
    open_trades = []
    closed_trades = []
    pair_cooldown = {}        # sym → round_idx when regular cooldown expires
    pair_sl_cooldown = {}     # sym → round_idx when SL cooldown expires (8 bars)
    consecutive_sl = 0
    global_pause = 0          # bars remaining

    for ri, rs in enumerate(rounds):
        rts = rs[0]['ts']
        hour = rts.hour

        # Tick down global pause (once per cycle)
        if global_pause > 0:
            global_pause -= 1

        # Step 1: Update & exit open trades
        trades_to_close = []
        prev_ts = rounds[ri - 1][0]['ts'] if ri > 0 else rts

        for trade in open_trades:
            ss = next((s for s in rs if s['sym'] == trade['sym']), None)
            if ss is None:
                trade['bars'] += 1
                continue

            trade['bars'] += 1

            # Use 1m candles if available for precise SL/TP/trailing
            if candles_1m and trade['sym'] in candles_1m:
                sub_candles = [c for c in candles_1m[trade['sym']]
                               if prev_ts < c['ts'] <= rts]
                for sc in sub_candles:
                    if trade['dir'] == 'LONG':
                        trade['best'] = max(trade['best'], sc['high'])
                    else:
                        trade['best'] = min(trade['best'], sc['low'])

                    sub_atr = ss['atr'] if ss else 0
                    if trail_enabled and sub_atr > 0:
                        act = sub_atr * trail_activation
                        dist = sub_atr * trail_distance
                        if trade['dir'] == 'LONG' and trade['best'] >= trade['entry'] + act:
                            trade['trail_active'] = True
                            new_sl = trade['best'] - dist
                            if new_sl > trade['csl']:
                                trade['csl'] = new_sl
                        elif trade['dir'] == 'SHORT' and trade['best'] <= trade['entry'] - act:
                            trade['trail_active'] = True
                            new_sl = trade['best'] + dist
                            if new_sl < trade['csl']:
                                trade['csl'] = new_sl

                    # Check TP on 1m
                    if trade['dir'] == 'LONG' and sc['high'] >= trade['tp']:
                        pnl = (trade['tp'] - trade['entry']) / trade['entry'] * 100 - 0.08
                        trade.update(reason='TP', pnl=pnl, et=sc['ts'])
                        trades_to_close.append(trade)
                        break
                    if trade['dir'] == 'SHORT' and sc['low'] <= trade['tp']:
                        pnl = (trade['entry'] - trade['tp']) / trade['entry'] * 100 - 0.08
                        trade.update(reason='TP', pnl=pnl, et=sc['ts'])
                        trades_to_close.append(trade)
                        break
                    # Check SL on 1m
                    if trade['dir'] == 'LONG' and sc['low'] <= trade['csl']:
                        r = 'TRAIL_SL' if trade['trail_active'] else 'SL'
                        pnl = (trade['csl'] - trade['entry']) / trade['entry'] * 100 - 0.08
                        trade.update(reason=r, pnl=pnl, et=sc['ts'])
                        trades_to_close.append(trade)
                        break
                    if trade['dir'] == 'SHORT' and sc['high'] >= trade['csl']:
                        r = 'TRAIL_SL' if trade['trail_active'] else 'SL'
                        pnl = (trade['entry'] - trade['csl']) / trade['entry'] * 100 - 0.08
                        trade.update(reason=r, pnl=pnl, et=sc['ts'])
                        trades_to_close.append(trade)
                        break

                if trade in trades_to_close:
                    continue

            hi, lo, cl = ss['high'], ss['low'], ss['price']
            atr = ss['atr']

            # Update best price
            if trade['dir'] == 'LONG':
                trade['best'] = max(trade['best'], hi)
            else:
                trade['best'] = min(trade['best'], lo)

            # Trailing SL
            if trail_enabled and atr > 0:
                act = atr * trail_activation
                dist = atr * trail_distance
                if trade['dir'] == 'LONG' and trade['best'] >= trade['entry'] + act:
                    trade['trail_active'] = True
                    new_sl = trade['best'] - dist
                    if new_sl > trade['csl']:
                        trade['csl'] = new_sl
                elif trade['dir'] == 'SHORT' and trade['best'] <= trade['entry'] - act:
                    trade['trail_active'] = True
                    new_sl = trade['best'] + dist
                    if new_sl < trade['csl']:
                        trade['csl'] = new_sl

            # TP check
            if trade['dir'] == 'LONG' and hi >= trade['tp']:
                pnl = (trade['tp'] - trade['entry']) / trade['entry'] * 100 - 0.08
                trade.update(reason='TP', pnl=pnl, et=rts)
                trades_to_close.append(trade)
                continue
            if trade['dir'] == 'SHORT' and lo <= trade['tp']:
                pnl = (trade['entry'] - trade['tp']) / trade['entry'] * 100 - 0.08
                trade.update(reason='TP', pnl=pnl, et=rts)
                trades_to_close.append(trade)
                continue

            # SL check (uses current_sl which may be trailing)
            if trade['dir'] == 'LONG' and lo <= trade['csl']:
                r = 'TRAIL_SL' if trade['trail_active'] else 'SL'
                pnl = (trade['csl'] - trade['entry']) / trade['entry'] * 100 - 0.08
                trade.update(reason=r, pnl=pnl, et=rts)
                trades_to_close.append(trade)
                continue
            if trade['dir'] == 'SHORT' and hi >= trade['csl']:
                r = 'TRAIL_SL' if trade['trail_active'] else 'SL'
                pnl = (trade['entry'] - trade['csl']) / trade['entry'] * 100 - 0.08
                trade.update(reason=r, pnl=pnl, et=rts)
                trades_to_close.append(trade)
                continue

            # TIMEOUT
            if trade['bars'] >= timeout:
                if trade['dir'] == 'LONG':
                    pnl = (cl - trade['entry']) / trade['entry'] * 100 - 0.08
                else:
                    pnl = (trade['entry'] - cl) / trade['entry'] * 100 - 0.08
                trade.update(reason='TIMEOUT', pnl=pnl, et=rts)
                trades_to_close.append(trade)
                continue

        # Process closed trades
        for t in trades_to_close:
            open_trades.remove(t)
            closed_trades.append(t)
            pair_cooldown[t['sym']] = ri + cooldown

            # Consecutive SL logic (TRAIL_SL with profit does NOT count)
            is_loss = t['reason'] == 'SL' or t['pnl'] < -0.2
            if is_loss:
                consecutive_sl += 1
                pair_sl_cooldown[t['sym']] = ri + 8  # 8 bars = 40 min
                if consecutive_sl >= 3:
                    global_pause = 24  # 2 hours
                    if verbose:
                        print(f"    PAUSE: {consecutive_sl} consecutive SL @ {rts}")
            else:
                consecutive_sl = 0

        # Step 2: Open new trades
        if global_pause > 0:
            continue
        if len(open_trades) >= max_open:
            continue
        if hour in dead_hours:
            continue

        # Dynamic threshold
        threshold = threshold_base
        if consecutive_sl >= 3:
            threshold += 0.15
        elif consecutive_sl >= 2:
            threshold += 0.075

        for ref in ref_by_round.get(ri, []):
            if len(open_trades) >= max_open:
                break
            if ref['conf'] < threshold:
                continue

            # Regular cooldown
            if ref['sym'] in pair_cooldown and ri < pair_cooldown[ref['sym']]:
                continue
            # SL cooldown
            if ref['sym'] in pair_sl_cooldown and ri < pair_sl_cooldown[ref['sym']]:
                continue
            # Already have position on this pair
            if any(t['sym'] == ref['sym'] for t in open_trades):
                continue
            # Regime filter
            if (ref['dir'], ref['regime']) in skip_regimes:
                continue

            # Disagreement penalty
            eff_conf = ref['conf']
            if ref['disagr'] >= 0.5:
                eff_conf *= 0.60
            elif ref['disagr'] > 0:
                eff_conf *= 0.85
            if eff_conf < threshold:
                continue

            # SL/TP from regime params
            params = regime_params.get(ref['regime'], {'sl': 1.5, 'tp': 3.0})
            sl_dist = ref['atr'] * params['sl']
            tp_dist = ref['atr'] * params['tp']
            min_sl = ref['entry'] * 0.3 / 100
            if sl_dist < min_sl:
                sl_dist = min_sl

            if ref['dir'] == 'LONG':
                sl, tp = ref['entry'] - sl_dist, ref['entry'] + tp_dist
            else:
                sl, tp = ref['entry'] + sl_dist, ref['entry'] - tp_dist

            trade = {
                'sym': ref['sym'], 'dir': ref['dir'],
                'entry': ref['entry'], 'sl': sl, 'tp': tp,
                'csl': sl, 'best': ref['entry'],
                'conf': eff_conf, 'regime': ref['regime'],
                'ot': rts, 'bars': 0, 'trail_active': False,
            }
            open_trades.append(trade)

            if verbose:
                print(f"  OPEN  {rts.strftime('%H:%M')} {ref['sym']:<12} {ref['dir']:<6} "
                      f"conf={eff_conf:.1%} regime={ref['regime']}")

    # Close remaining open trades at last price
    for t in open_trades:
        last = sym_scans[t['sym']][-1]
        if t['dir'] == 'LONG':
            pnl = (last['price'] - t['entry']) / t['entry'] * 100 - 0.08
        else:
            pnl = (t['entry'] - last['price']) / t['entry'] * 100 - 0.08
        t.update(reason='STILL_OPEN', pnl=pnl, et=last['ts'])
        closed_trades.append(t)

    return closed_trades


# ── Output ────────────────────────────────────────────────────

def print_results(name, trades):
    cl = [t for t in trades if t['reason'] != 'STILL_OPEN']
    if not cl:
        print(f"  {name:<45} | 0 trades")
        return

    w = sum(1 for t in cl if t['pnl'] > 0)
    p = sum(t['pnl'] for t in cl)
    lo = [t for t in cl if t['dir'] == 'LONG']
    sh = [t for t in cl if t['dir'] == 'SHORT']
    lp = sum(t['pnl'] for t in lo)
    sp = sum(t['pnl'] for t in sh)
    tp = sum(1 for t in cl if t['reason'] == 'TP')
    sl = sum(1 for t in cl if t['reason'] in ('SL', 'TRAIL_SL'))
    to = sum(1 for t in cl if t['reason'] == 'TIMEOUT')
    ee = sum(1 for t in cl if t['reason'] == 'EARLY_EXIT')

    parts = [f"TP:{tp}", f"SL:{sl}"]
    if to:
        parts.append(f"TO:{to}")
    if ee:
        parts.append(f"EE:{ee}")

    print(f"  {name:<45} | {len(cl):>2}t WR {w}/{len(cl)}={w/len(cl)*100:>3.0f}% "
          f"| PnL {p:>+6.2f}% | L:{len(lo)}({lp:>+5.2f}%) S:{len(sh)}({sp:>+5.2f}%) "
          f"| {' '.join(parts)}")


def print_trades(trades):
    cl = [t for t in trades if t['reason'] != 'STILL_OPEN']
    for i, t in enumerate(cl, 1):
        print(f"    {i:>2}. {t['ot'].strftime('%H:%M')} {t['sym']:<12} {t['dir']:<6} "
              f"conf={t['conf']:.1%} regime={t['regime']:<9} "
              f"| {t['reason']:<9} {t['pnl']:>+6.2f}%")


# ── Main ──────────────────────────────────────────────────────

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/anatolii/Desktop/trades_v2.log'
    paper_path = sys.argv[2] if len(sys.argv) > 2 else '/Users/anatolii/Desktop/paper_trades_v2.json'

    scans, refineds, refined_invs, candles_1m = parse_logs(log_path)
    sym_scans, rounds, ref_by_round, inv_by_round = build_structures(scans, refineds, refined_invs, candles_1m)

    c1m_count = sum(len(v) for v in candles_1m.values())

    ref_l = sum(1 for r in refineds if r['dir'] == 'LONG')
    ref_s = sum(1 for r in refineds if r['dir'] == 'SHORT')
    print(f"Parsed: {len(scans)} SCAN, {len(refineds)} REFINED (L:{ref_l} S:{ref_s}), "
          f"{c1m_count} CANDLE_1M, {len(rounds)} rounds")
    print(f"Time: {rounds[0][0]['ts']} → {rounds[-1][0]['ts']}")
    print()

    # Paper results
    print("=" * 140)
    print("PAPER (реальність):")
    print("=" * 140)
    with open(paper_path) as f:
        real = json.load(f)
    rc = [t for t in real if t['status'] == 'CLOSED']
    if rc:
        rw = sum(1 for t in rc if t['pnl_pct'] > 0)
        rp = sum(t['pnl_pct'] for t in rc)
        rl = [t for t in rc if t['direction'] == 'LONG']
        rs = [t for t in rc if t['direction'] == 'SHORT']
        print(f"  {'PAPER':<45} | {len(rc):>2}t WR {rw}/{len(rc)}={rw/len(rc)*100:>3.0f}% "
              f"| PnL {rp:>+6.2f}% | L:{len(rl)}({sum(t['pnl_pct'] for t in rl):>+5.2f}%) "
              f"S:{len(rs)}({sum(t['pnl_pct'] for t in rs):>+5.2f}%)")

    # Verification
    print()
    print("=" * 140)
    print("ПЕРЕВІРКА (має збігатись з paper):")
    print("=" * 140)
    current = simulate(rounds, sym_scans, ref_by_round, {}, candles_1m=candles_1m, verbose=False)
    print_results("Current config", current)
    print("  Трейди:")
    print_trades(current)

    # Per-regime SL/TP
    print()
    print("=" * 140)
    print("PER-REGIME SL/TP:")
    print("=" * 140)
    for name, t_p, r_p, rv_p in [
        ("Flat TP2.0", {'sl':1.5,'tp':2.0}, {'sl':1.5,'tp':2.0}, {'sl':1.5,'tp':2.0}),
        ("Flat TP1.5", {'sl':1.5,'tp':1.5}, {'sl':1.5,'tp':1.5}, {'sl':1.5,'tp':1.5}),
        ("RANGE:TP1.5 rest:TP3.0", {'sl':1.5,'tp':3.0}, {'sl':1.5,'tp':1.5}, {'sl':1.5,'tp':3.0}),
        ("RANGE:TP2.0 rest:TP3.0", {'sl':1.5,'tp':3.0}, {'sl':1.5,'tp':2.0}, {'sl':1.5,'tp':3.0}),
        ("T:TP2.0 R:TP1.5 REV:TP2.0", {'sl':1.5,'tp':2.0}, {'sl':1.5,'tp':1.5}, {'sl':1.5,'tp':2.0}),
    ]:
        print_results(name, simulate(rounds, sym_scans, ref_by_round,
            {'regime_params': {'TREND': t_p, 'RANGE': r_p, 'REVERSAL': rv_p}}, candles_1m=candles_1m))

    # Trailing
    print()
    print("=" * 140)
    print("TRAILING SL:")
    print("=" * 140)
    for a in [1.0, 1.5, 2.0, 2.5, 3.0]:
        print_results(f"Trail activation={a}x", simulate(rounds, sym_scans, ref_by_round,
            {'trail_activation': a}, candles_1m=candles_1m))
    print_results("Trail DISABLED", simulate(rounds, sym_scans, ref_by_round,
        {'trail_enabled': False}, candles_1m=candles_1m))

    # Confidence
    print()
    print("=" * 140)
    print("CONFIDENCE:")
    print("=" * 140)
    for thr in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        print_results(f"Conf >= {thr:.0%}", simulate(rounds, sym_scans, ref_by_round,
            {'threshold': thr}, candles_1m=candles_1m))

    # Timeout
    print()
    print("=" * 140)
    print("TIMEOUT:")
    print("=" * 140)
    for to in [6, 12, 18, 24]:
        print_results(f"Timeout={to} ({to*5}min)", simulate(rounds, sym_scans, ref_by_round,
            {'timeout': to}, candles_1m=candles_1m))

    # Regime filter
    print()
    print("=" * 140)
    print("REGIME FILTER:")
    print("=" * 140)
    print_results("Skip LONG in REVERSAL", simulate(rounds, sym_scans, ref_by_round,
        {'skip_regimes': [('LONG', 'REVERSAL')]}, candles_1m=candles_1m))
    print_results("Skip LONG in RANGE", simulate(rounds, sym_scans, ref_by_round,
        {'skip_regimes': [('LONG', 'RANGE')]}, candles_1m=candles_1m))

    # Best combos
    print()
    print("=" * 140)
    print("BEST COMBOS:")
    print("=" * 140)
    for name, cfg in [
        ("conf75% + trail2.0", {'threshold': 0.75, 'trail_activation': 2.0}),
        ("conf75% + trail2.0 + skipL/REV", {'threshold': 0.75, 'trail_activation': 2.0,
            'skip_regimes': [('LONG', 'REVERSAL')]}),
        ("conf80% + trail2.5", {'trail_activation': 2.5}),
        ("conf80% + trail2.5 + skipL/REV", {'trail_activation': 2.5,
            'skip_regimes': [('LONG', 'REVERSAL')]}),
    ]:
        print_results(name, simulate(rounds, sym_scans, ref_by_round, cfg, candles_1m=candles_1m))

    # Inverse
    print()
    print("=" * 140)
    print("ІНВЕРСНА ТОРГІВЛЯ:")
    print("=" * 140)
    print_results("Inverse current", simulate(rounds, sym_scans, inv_by_round, {}, candles_1m=candles_1m))
    print_results("Inverse TP1.5", simulate(rounds, sym_scans, inv_by_round, {
        'regime_params': {'TREND': {'sl':1.5,'tp':1.5}, 'RANGE': {'sl':1.5,'tp':1.5},
                          'REVERSAL': {'sl':1.5,'tp':1.5}}}, candles_1m=candles_1m))


if __name__ == "__main__":
    main()
