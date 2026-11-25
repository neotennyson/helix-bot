import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Helix | Research Terminal", page_icon=None, layout="wide")


def load_json(path: str) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def money(value: Any) -> str:
    try:
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


def pct(value: Any, signed: bool = False) -> str:
    try:
        number = float(value) * 100
        return f"{'+' if signed and number > 0 else ''}{number:.2f}%"
    except (TypeError, ValueError):
        return "—"


def num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def tone(value: Any, higher_is_good: bool = True) -> str:
    try:
        good = float(value) >= 0 if higher_is_good else float(value) > 0
        return "positive" if good else "negative"
    except (TypeError, ValueError):
        return "neutral"


def pill(text: str, level: str = "neutral") -> str:
    return f'<span class="pill pill-{esc(level)}">{esc(text)}</span>'


def card(label: str, value: str, note: str = "", level: str = "neutral") -> str:
    note_html = f'<div class="card-note">{esc(note)}</div>' if note else ""
    return f'<div class="kpi-card tone-{esc(level)}"><div class="card-label">{esc(label)}</div><div class="card-value">{esc(value)}</div>{note_html}</div>'


def section(label: str, title: str, note: str = "") -> str:
    note_html = f'<div class="section-note">{esc(note)}</div>' if note else ""
    return f'<div class="section-head"><div class="eyebrow">{esc(label)}</div><h2>{esc(title)}</h2>{note_html}</div>'


def panel(title: str, note: str = "") -> str:
    note_html = f'<span class="panel-note">{esc(note)}</span>' if note else ""
    return f'<div class="panel-head"><span class="panel-title">{esc(title)}</span>{note_html}</div>'


def show_table(frame: pd.DataFrame | pd.io.formats.style.Styler, height: int = 280) -> None:
    data = frame.data if isinstance(frame, pd.io.formats.style.Styler) else frame
    if data.empty:
        st.info("No rows available.")
        return
    st.dataframe(frame, width="stretch", height=height, hide_index=True)


def context() -> tuple[dict | None, dict | None, dict | None, dict | None]:
    return (load_json("reports/backtest.json"), load_json("reports/optimization.json"), load_json("reports/research.json"), load_json("data/paper_state.json"))


def universe_view() -> None:
    view_header("Universe", "Multi-instrument screen across stocks, indices and options. Screening is in-sample; the holdout gate is the verdict.", pill("SCREEN ONLY", "amber"))
    report = load_json("reports/universe_scan.json")
    if report is None:
        st.info("No universe scan found. Run: python3 -m helix.cli universe --profiles config/equity_profiles.json")
        return
    portfolio = report.get("portfolio", {}).get("metrics", {})
    qualifying = report.get("qualifying_symbols", [])
    rules = report.get("promotion_rules", {})
    minimum_sharpe = rules.get("minimum_sharpe", 0.5)
    profile_count = len(report.get("profile_summary", []))
    st.markdown(
        '<div class="kpi-grid">'
        + card("Symbols scanned", str(report.get("universe_size", "—")), f"{len(equity_files())} csv on disk", "neutral")
        + card("Config scans", str(len(report.get("results", {}))), f"{profile_count} mode/profile combinations", "neutral")
        + card("Qualifying", str(len(qualifying)), f"sharpe >= {minimum_sharpe}", "green" if qualifying else "amber")
        + card("Portfolio CAGR", pct(portfolio.get("cagr")), "equal weight, shortlist", tone(portfolio.get("cagr")))
        + card("Portfolio Sharpe", num(portfolio.get("sharpe")), "shortlist blend", "positive" if float(portfolio.get("sharpe") or 0) > 0.5 else "amber")
        + card("Max drawdown", pct(portfolio.get("max_drawdown")), "shortlist blend", "positive" if float(portfolio.get("max_drawdown") or 0) > -0.15 else "negative")
        + "</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([3, 2])
    with left:
        st.markdown(panel("Profile summary", "median across the scanned universe"), unsafe_allow_html=True)
        profiles = report.get("profile_summary", [])
        if profiles:
            show_table(
                pd.DataFrame(
                    [
                        {
                            "mode": row["strategy_mode"],
                            "profile": row["profile"],
                            "median return": row["median_total_return"],
                            "median sharpe": row["median_sharpe"],
                            "best sharpe": row["best_sharpe"],
                            "median trades": row["median_trades"],
                            "in profit": f"{row['symbols_in_profit']}/{row['symbols']}",
                        }
                        for row in profiles
                    ]
                ).style.format(
                    {
                        "median return": "{:+.2%}",
                        "median sharpe": "{:.3f}",
                        "best sharpe": "{:.3f}",
                        "median trades": "{:.0f}",
                    }
                ),
                height=320,
            )
    with right:
        st.markdown(panel("Buy and hold reference", "beta, not a strategy"), unsafe_allow_html=True)
        reference = report.get("buy_and_hold_reference", {})
        st.markdown(
            f'<div class="panel-body">'
            f'<div class="row"><span>Mean first-to-last</span><strong>{pct(reference.get("mean_total_return"))}</strong></div>'
            f'<div class="row"><span>Median first-to-last</span><strong>{pct(reference.get("median_total_return"))}</strong></div>'
            f'<div class="note">{esc(reference.get("note", ""))}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(panel("Portfolio blend", "shortlist sleeves"), unsafe_allow_html=True)
        st.markdown(
            f'<div class="panel-body">'
            f'<div class="row"><span>Weighting</span><strong>{esc(report.get("portfolio", {}).get("weighting", "—"))}</strong></div>'
            f'<div class="row"><span>Trades</span><strong>{esc(portfolio.get("trades"))}</strong></div>'
            f'<div class="row"><span>Avg pairwise corr</span><strong>{num(portfolio.get("average_pairwise_correlation"), 3)}</strong></div>'
            f'<div class="row"><span>Sleeves</span><strong>{esc(portfolio.get("sleeves"))}</strong></div>'
            f'<div class="row"><span>Profitable share</span><strong>{pct(portfolio.get("profitable_symbol_share"))}</strong></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown(panel("Ranking", "best configuration per symbol, in-sample"), unsafe_allow_html=True)
    rows = report.get("ranking", [])
    if rows:
        frame = pd.DataFrame(rows).head(60)
        show_table(
            frame[["symbol", "strategy_mode", "profile", "total_return", "cagr", "sharpe", "max_drawdown", "profit_factor", "trades", "qualifies"]].style.format(
                {"total_return": "{:+.2%}", "cagr": "{:+.2%}", "sharpe": "{:.3f}", "max_drawdown": "{:.2%}"}
            ),
            height=420,
        )
    st.caption("Screening uses full history, so it is not an out-of-sample result. Only reports/equity_research_*.json holdout numbers can decide promotion.")


def equity_files() -> list:
    return sorted(Path("data/equity").glob("*_daily.csv"))


def options_view() -> None:
    view_header("Options", "Modelled index-option premium strategies. No real option quotes are available from the broker adapters.", pill("SYNTHETIC PRICING", "amber"))
    report = load_json("reports/options_research.json")
    if report is None:
        st.info("No options research found. Run: python3 -m helix.cli options-research --data data/nifty_fut_oct2026_daily.csv --symbol 48704")
        return
    st.warning(
        f"{report.get('disclaimer', '')} Underlying path: {report.get('underlying_symbol')} "
        f"({report.get('underlying_bars')} real daily bars). Any result here is modelled, not quoted."
    )
    results = report.get("results", [])
    st.markdown(
        f'<div class="kpi-grid">'
        f'{card("Configurations", str(report.get("runs", "—")), "strategy x iv x delta x holding", "neutral")}'
        f'{card("Best Sharpe", num(report.get("best_sharpe")), "modelled", tone(report.get("best_sharpe"), False))}'
        f'{card("Promotion", "BLOCKED", "synthetic pricing cannot qualify", "amber")}'
        f'{card("Real quotes", "NO", "broker adapters return none", "amber")}'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(panel("Configuration grid", "ranked by modelled Sharpe"), unsafe_allow_html=True)
    if results:
        frame = pd.DataFrame(
            [
                {
                    "strategy": row["grid"]["strategy"],
                    "iv": row["grid"]["implied_volatility"],
                    "delta": row["grid"]["delta_target"],
                    "holding days": row["grid"]["holding_days"],
                    "cagr": row["metrics"].get("cagr"),
                    "sharpe": row["metrics"].get("sharpe"),
                    "max dd": row["metrics"].get("max_drawdown"),
                    "trades": row["metrics"].get("trades"),
                    "win rate": row["metrics"].get("win_rate"),
                    "excess over cash": row.get("excess_cagr_over_cash"),
                }
                for row in results
            ]
        )
        show_table(
            frame.style.format(
                {
                    "iv": "{:.2f}",
                    "delta": "{:.2f}",
                    "cagr": "{:+.2%}",
                    "sharpe": "{:.3f}",
                    "max dd": "{:.2%}",
                    "win rate": "{:.1%}",
                    "excess over cash": "{:+.2%}",
                }
            ),
            height=420,
        )
    st.caption("Modelled Sharpe rises with assumed IV because the engine holds implied vol constant. Real short premium carries vega and spread risk this model omits.")


def main() -> None:
    backtest, optimization, research, paper_state = context()
    page = shell("Overview", backtest, paper_state)
    if page == "Overview":
        overview(backtest, research, paper_state)
    elif page == "Validation":
        validation(optimization, research)
    elif page == "Universe":
        universe_view()
    elif page == "Options":
        options_view()
    elif page == "Paper":
        paper(paper_state)
    else:
        trades(backtest)
    st.markdown('<div class="bottom-line"></div><div class="bottom-footer"><span>HELIX</span><span>Read-only research terminal</span><span>Powered by Space Bunny</span></div>', unsafe_allow_html=True)


def shell(page: str, backtest: dict | None, paper: dict | None) -> str:
    source = backtest.get("data_source", "no data") if backtest else "no data"
    mode = backtest.get("config", {}).get("strategy_mode", "—") if backtest else "—"
    last_bar = "—"
    if paper and paper.get("last_timestamp"):
        last_bar = str(paper["last_timestamp"]).replace("T", " ")[:16]
    elif backtest and backtest.get("equity_curve"):
        last_bar = str(backtest["equity_curve"][-1]["timestamp"]).replace("T", " ")[:16]
    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand"><span class="brand-mark">H</span><span>HELIX</span><span class="brand-divider">/</span><span class="muted">RESEARCH TERMINAL</span></div>
          <div class="topbar-right">{pill('PAPER ONLY', 'green')}<span class="muted">{esc(source)}</span><span class="faint">LAST BAR {esc(last_bar)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pages = ["Overview", "Validation", "Universe", "Options", "Paper", "Trades"]
    with st.sidebar:
        st.markdown('<div class="side-brand"><span class="brand-mark">H</span><div><strong>HELIX</strong><small>F&O RESEARCH OPS</small></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="side-label">WORKSPACE</div>', unsafe_allow_html=True)
        selected = st.radio("Workspace", pages, index=pages.index(page) if page in pages else 0, label_visibility="collapsed", key="workspace_nav")
        st.markdown('<div class="side-label">SESSION</div>', unsafe_allow_html=True)
        trades = backtest.get("metrics", {}).get("trades", 0) if backtest else 0
        st.markdown(f'<div class="side-status"><div><i class="dot green"></i>Data connected</div><div><i class="dot amber"></i>Paper engine</div><div><i class="dot blue"></i>{esc(trades)} backtest trades</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="side-foot">MODEL<br><span class="muted">{esc(mode)}</span><br><br>Read-only surface<br>Powered by Space Bunny</div>', unsafe_allow_html=True)
    return selected


def view_header(title: str, subtitle: str, right: str = "") -> None:
    st.markdown(f'<div class="view-head"><div><div class="eyebrow">HELIX / OPERATIONS</div><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div><div class="view-head-right">{right}</div></div>', unsafe_allow_html=True)


def overview(backtest: dict | None, research: dict | None, paper: dict | None) -> None:
    if backtest is None:
        st.info("No backtest report found.")
        return
    metrics = backtest.get("metrics", {})
    equity = backtest.get("final_equity")
    capital = backtest.get("initial_capital")
    pnl = equity - capital if isinstance(equity, (int, float)) and isinstance(capital, (int, float)) else None
    hurdle = backtest.get("config", {}).get("risk", {}).get("cash_hurdle_annual", 0.06)
    excess_cagr = (metrics.get("cagr") or 0.0) - hurdle
    view_header("Overview", "The short version: performance, risk, and what needs attention.", pill("PAPER / READ-ONLY", "green"))
    st.markdown(f'<div class="kpi-grid">{card("Equity", money(equity), f"net {money(pnl)}", tone(pnl))}{card("Return", pct(metrics.get("total_return"), True), f"excess CAGR {pct(excess_cagr, True)}", tone(metrics.get("total_return")))}{card("Drawdown", pct(metrics.get("max_drawdown")), "peak to trough", "positive" if (metrics.get("max_drawdown") or 0) > -0.15 else "negative")}{card("Profit factor", num(metrics.get("profit_factor")), f"{pct(metrics.get("win_rate"))} win rate", tone(metrics.get("profit_factor"), False))}</div>', unsafe_allow_html=True)
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.markdown(panel("Equity curve", "historical account value"), unsafe_allow_html=True)
        curve = pd.DataFrame(backtest.get("equity_curve", []))
        if curve.empty:
            st.info("No equity curve available.")
        else:
            curve["timestamp"] = pd.to_datetime(curve["timestamp"])
            st.line_chart(curve.set_index("timestamp")[["equity"]], height=315)
    with right:
        st.markdown(panel("Risk tape", "the numbers that matter"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            ["CAGR", pct(metrics.get("cagr"), True)],
            ["Volatility", pct(metrics.get("annualized_volatility"))],
            ["Sharpe", num(metrics.get("sharpe"))],
            ["Expectancy", money(metrics.get("expectancy"))],
            ["Fees", money(metrics.get("total_fees"))],
            ["Exposure", pct(metrics.get("exposure_fraction"))],
        ], columns=["Measure", "Value"]), width="stretch", hide_index=True, height=315)
    bottom_left, bottom_right = st.columns([1, 1], gap="large")
    with bottom_left:
        st.markdown(panel("Validation status", "is the candidate ready?"), unsafe_allow_html=True)
        if research:
            holdout = research.get("validation", {}).get("holdout_metrics", {})
            strict = research.get("validation", {}).get("promotion", {}).get("status", "unknown")
            st.markdown(f'<div class="status-line">{pill("PAPER REVIEW CANDIDATE" if research.get("paper_review_candidate") else "RESEARCH ONLY", "green" if research.get("paper_review_candidate") else "amber")}<span class="muted">strict gate: {esc(strict)}</span></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                ["Holdout return", pct(holdout.get("total_return"), True)],
                ["Holdout PF", num(holdout.get("profit_factor"))],
                ["Holdout drawdown", pct(holdout.get("max_drawdown"))],
                ["Holdout trades", holdout.get("trades", 0)],
            ], columns=["Check", "Value"]), width="stretch", hide_index=True, height=190)
    with bottom_right:
        st.markdown(panel("Paper account", "current simulator state"), unsafe_allow_html=True)
        if paper:
            st.markdown(f'<div class="status-line">{pill(str(paper.get("status", "unknown")).upper(), "green" if paper.get("status") == "flat" else "amber")}<span class="muted">{len(paper.get("trades", []))} closed trades</span></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                ["Equity", money(paper.get("equity"))],
                ["Cash", money(paper.get("cash"))],
                ["Position", "Open" if paper.get("position") else "Flat"],
                ["Broker", paper.get("broker", "none")],
            ], columns=["Account", "Value"]), width="stretch", hide_index=True, height=190)
        else:
            st.info("No paper state found.")
    with st.expander("Monthly returns"):
        monthly = metrics.get("monthly_returns", {})
        if monthly:
            st.bar_chart(pd.DataFrame({"month": list(monthly), "return": list(monthly.values())}).set_index("month"), height=210)
        else:
            st.info("No monthly history available.")


def validation(optimization: dict | None, research: dict | None) -> None:
    if optimization is None:
        st.info("No validation report found.")
        return
    view_header("Validation", "Separate the pretty result from the result that survived testing.", pill("HOLDOUT UNTOUCHED", "blue"))
    holdout = optimization.get("holdout_metrics", {})
    strict = optimization.get("promotion", {})
    view_status = "PAPER REVIEW CANDIDATE" if research and research.get("paper_review_candidate") else "RESEARCH ONLY"
    st.markdown(f'<div class="kpi-grid">{card("Holdout return", pct(holdout.get("total_return"), True), f"{holdout.get("trades", 0)} trades", tone(holdout.get("total_return")))}{card("Profit factor", num(holdout.get("profit_factor")), "gross wins / gross losses", tone(holdout.get("profit_factor"), False))}{card("Drawdown", pct(holdout.get("max_drawdown")), "holdout peak to trough", "positive" if (holdout.get("max_drawdown") or 0) > -0.15 else "negative")}{card("Gate", "OPEN" if research and research.get("paper_review_candidate") else "PENDING", view_status, "green" if research and research.get("paper_review_candidate") else "amber")}</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown(panel("Selected model", "effective parameters"), unsafe_allow_html=True)
        st.json(optimization.get("selected_parameters", {}))
    with right:
        st.markdown(panel("Validation folds", "each fold is a separate time slice"), unsafe_allow_html=True)
        rows = []
        for fold in optimization.get("folds", []):
            metrics = fold.get("validation_metrics", {})
            rows.append({"fold": fold.get("fold"), "window": f"{str(fold.get('validation_start', ''))[:10]} → {str(fold.get('validation_end', ''))[:10]}", "return": pct(metrics.get("total_return"), True), "drawdown": pct(metrics.get("max_drawdown")), "trades": metrics.get("trades", 0)})
        show_table(pd.DataFrame(rows), 250)
    if research:
        st.markdown(section("ROBUSTNESS", "Rolling windows", "A single good holdout is not enough."), unsafe_allow_html=True)
        rows = []
        for window in research.get("rolling_windows", []):
            rows.append({"window": f"{str(window.get('start', ''))[:10]} → {str(window.get('end', ''))[:10]}", "return": pct(window.get("total_return"), True), "PF": num(window.get("profit_factor")), "drawdown": pct(window.get("max_drawdown")), "trades": window.get("trades", 0)})
        show_table(pd.DataFrame(rows), 220)
        st.markdown(panel("Cost stress", "fees and slippage increased"), unsafe_allow_html=True)
        rows = []
        for name, result in research.get("cost_stress", {}).items():
            metrics = result.get("metrics", {})
            rows.append({"scenario": name.upper(), "return": pct(metrics.get("total_return"), True), "PF": num(metrics.get("profit_factor")), "drawdown": pct(metrics.get("max_drawdown")), "trades": result.get("trades", 0)})
        show_table(pd.DataFrame(rows), 190)
    st.caption(f"Strict promotion status: {strict.get('status', 'unknown')}. Cash hurdle: {pct(research.get('cash_hurdle_annual') if research else 0.06)}. A paper-review candidate is not a live-trading approval.")


def paper(paper_state: dict | None) -> None:
    view_header("Paper", "Simulated account state. No broker order path is enabled.", pill("PAPER ONLY", "green"))
    if paper_state is None:
        st.info("No paper state found.")
        return
    metrics = paper_state.get("metrics", {})
    st.markdown(f'<div class="kpi-grid">{card("Equity", money(paper_state.get("equity")), "mark to market", tone((paper_state.get("equity") or 0) - (paper_state.get("initial_capital") or 0)))}{card("Cash", money(paper_state.get("cash")), "settled balance", "neutral")}{card("Trades", str(len(paper_state.get("trades", []))), "closed simulator trades", "neutral")}{card("State", str(paper_state.get("status", "unknown")).upper(), paper_state.get("broker", "none"), "green")}</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown(panel("Position", "current exposure"), unsafe_allow_html=True)
        if paper_state.get("position"):
            st.json(paper_state["position"])
        else:
            st.markdown('<div class="empty-state">No open position</div>', unsafe_allow_html=True)
        if paper_state.get("pending"):
            st.warning("Next-bar decision staged")
            st.json(paper_state["pending"])
    with right:
        st.markdown(panel("Paper equity", "simulator curve"), unsafe_allow_html=True)
        curve = pd.DataFrame(paper_state.get("equity_curve", []))
        if curve.empty:
            st.info("No curve available.")
        else:
            curve["timestamp"] = pd.to_datetime(curve["timestamp"])
            st.line_chart(curve.set_index("timestamp")[["equity"]], height=280)
    st.markdown(panel("Recent paper trades", "last executed simulations"), unsafe_allow_html=True)
    trades = pd.DataFrame(paper_state.get("trades", []))
    if not trades.empty:
        show_table(trades, 280)
    else:
        st.info("No paper trades yet.")


def trades(backtest: dict | None) -> None:
    view_header("Trades", "Filter the historical executions and inspect the reason for every exit.", pill("READ ONLY", "blue"))
    if backtest is None or not backtest.get("trades"):
        st.info("No trade history found.")
        return
    frame = pd.DataFrame(backtest["trades"])
    controls = st.columns([1, 1, 3])
    with controls[0]:
        side = st.selectbox("Side", ["ALL", "LONG", "SHORT"])
    with controls[1]:
        reason = st.selectbox("Exit", ["ALL"] + sorted(frame["exit_reason"].dropna().unique().tolist()))
    filtered = frame
    if side != "ALL":
        filtered = filtered[filtered["side"] == side]
    if reason != "ALL":
        filtered = filtered[filtered["exit_reason"] == reason]
    filtered = filtered.copy()
    filtered["entry_time"] = pd.to_datetime(filtered["entry_time"])
    filtered["exit_time"] = pd.to_datetime(filtered["exit_time"])
    filtered["net_pnl"] = pd.to_numeric(filtered["net_pnl"], errors="coerce")
    net = filtered["net_pnl"].sum()
    st.markdown(f'<div class="kpi-grid">{card("Visible trades", str(len(filtered)), "current filter", "neutral")}{card("Net P&L", money(net), "after modeled costs", tone(net))}{card("Win rate", pct((filtered["net_pnl"] > 0).mean() if not filtered.empty else None), "closed trades", "neutral")}{card("Average hold", f"{num(filtered["bars_held"].mean(), 1) if not filtered.empty else "—"} bars", "time in market", "neutral")}</div>', unsafe_allow_html=True)
    columns = ["entry_time", "exit_time", "side", "quantity", "entry_price", "exit_price", "gross_pnl", "fees", "net_pnl", "exit_reason", "entry_reason"]
    show_table(filtered[columns].sort_values("entry_time", ascending=False), 560)
    st.download_button("Download blotter", filtered[columns].to_csv(index=False).encode("utf-8"), file_name="helix_blotter.csv", mime="text/csv")


CSS = """
<style>
:root { --bg:#0b0e13; --panel:#12171f; --panel-2:#161c25; --line:#28313d; --text:#e8edf3; --muted:#8b96a5; --faint:#5c6876; --green:#45c9a0; --red:#ef6b78; --amber:#d6a75d; --blue:#79a9e6; }
html,body,[class*="css"] { font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.stMain,[data-testid="stMain"] { background: var(--bg); }
.block-container { max-width: 1480px; padding: 1rem 2.2rem 2.5rem; }
[data-testid="stHeader"] { background: var(--bg); border-bottom: 1px solid var(--line); }
#MainMenu,footer { visibility:hidden; }
h1,h2,h3 { letter-spacing:-.025em; }
h1 { margin:0; font-size:2rem; line-height:1.05; }
h2 { margin:0.25rem 0 0; font-size:1.15rem; }
p { margin:0.45rem 0 0; color:var(--muted); font-size:.88rem; }
.topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.2rem 0 1rem; border-bottom:1px solid var(--line); }
.brand,.topbar-right,.side-brand,.side-status div { display:flex; align-items:center; gap:.55rem; }
.brand { font-size:.72rem; font-weight:800; letter-spacing:.13em; }
.brand-mark { display:inline-flex; align-items:center; justify-content:center; width:1.55rem; height:1.55rem; border:1px solid var(--green); border-radius:.3rem; color:var(--green); font-size:.62rem; font-weight:800; }
.brand-divider { color:var(--faint); margin:0 .15rem; }
.topbar-right { font-size:.66rem; }
.muted { color:var(--muted); }
.faint { color:var(--faint); }
.pill { display:inline-flex; align-items:center; border-radius:999px; padding:.22rem .5rem; font-size:.58rem; font-weight:800; letter-spacing:.08em; }
.pill-green { color:var(--green); background:rgba(69,201,160,.11); border:1px solid rgba(69,201,160,.25); }
.pill-amber { color:var(--amber); background:rgba(214,167,93,.11); border:1px solid rgba(214,167,93,.25); }
.pill-blue { color:var(--blue); background:rgba(121,169,230,.11); border:1px solid rgba(121,169,230,.25); }
.pill-neutral { color:var(--muted); background:rgba(139,150,165,.1); border:1px solid var(--line); }
.view-head { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; padding:1.7rem 0 1.35rem; }
.view-head p { max-width:650px; }
.eyebrow,.side-label,.card-label,.panel-title { color:var(--faint); font-size:.62rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
.view-head-right { padding-bottom:.15rem; }
.kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; margin:.2rem 0 1.6rem; }
.kpi-card { min-height:104px; padding:.85rem .95rem; border:1px solid var(--line); border-radius:.4rem; background:var(--panel); }
.card-value { margin-top:.55rem; font-family:"JetBrains Mono","SFMono-Regular",Consolas,monospace; font-size:1.45rem; font-weight:700; letter-spacing:-.04em; font-variant-numeric:tabular-nums; }
.card-note { margin-top:.35rem; color:var(--muted); font-size:.72rem; }
.tone-positive .card-value { color:var(--green); }
.tone-negative .card-value { color:var(--red); }
.tone-neutral .card-value { color:var(--text); }
.section-head { margin:1.5rem 0 .75rem; }
.section-note { margin-top:.25rem; color:var(--muted); font-size:.8rem; }
.panel-head { display:flex; align-items:baseline; justify-content:space-between; gap:.7rem; margin-bottom:.5rem; }
.panel-title { color:var(--text); letter-spacing:.08em; }
.panel-note { color:var(--faint); font-size:.7rem; }
.status-line { display:flex; align-items:center; gap:.55rem; margin:.2rem 0 .7rem; font-size:.78rem; }
.empty-state { display:flex; align-items:center; justify-content:center; min-height:150px; border:1px dashed var(--line); border-radius:.35rem; color:var(--muted); }
section[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid var(--line); }
.side-brand { padding:.2rem 0 1rem; border-bottom:1px solid var(--line); }
.side-brand strong { display:block; font-size:.75rem; letter-spacing:.12em; }
.side-brand small { display:block; margin-top:.15rem; color:var(--faint); font-size:.57rem; letter-spacing:.1em; }
.side-label { margin:1.25rem 0 .5rem; }
.side-status { display:flex; flex-direction:column; gap:.55rem; color:var(--muted); font-size:.73rem; }
.dot { display:inline-block; width:.42rem; height:.42rem; margin-top:.18rem; border-radius:50%; }
.dot.green { background:var(--green); }.dot.amber { background:var(--amber); }.dot.blue { background:var(--blue); }
.side-foot { margin-top:2rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--faint); font-size:.64rem; line-height:1.65; letter-spacing:.06em; }
section[data-testid="stSidebar"] [role="radiogroup"] { gap:.2rem; }
section[data-testid="stSidebar"] [role="radiogroup"] label { padding:.45rem .55rem; border:1px solid transparent; border-radius:.3rem; color:var(--muted); }
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(255,255,255,.035); color:var(--text); }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:rgba(69,201,160,.08); border-color:rgba(69,201,160,.2); color:var(--green); }
.stButton>button,.stDownloadButton>button { border:1px solid var(--line); border-radius:.3rem; background:var(--panel-2); color:var(--text); }
.stButton>button:hover,.stDownloadButton>button:hover { border-color:var(--green); color:var(--green); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:.35rem; overflow:hidden; }
[data-testid="stDataFrame"] * { font-family:"JetBrains Mono","SFMono-Regular",Consolas,monospace; font-variant-numeric:tabular-nums; }
[data-testid="stAlert"] { border:1px solid rgba(214,167,93,.25); background:rgba(214,167,93,.07); color:var(--amber); }
@media (max-width:1000px) { .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .view-head { align-items:flex-start; flex-direction:column; } }
@media (max-width:640px) { .block-container { padding:.75rem 1rem 2rem; } .kpi-grid { grid-template-columns:1fr; } .topbar { align-items:flex-start; flex-direction:column; } .topbar-right { flex-wrap:wrap; } }
"""

st.markdown(CSS, unsafe_allow_html=True)


st.markdown('<style>.bottom-line{height:1px;margin-top:2rem;background:var(--line)}.bottom-footer{display:flex;justify-content:space-between;padding-top:.7rem;color:var(--faint);font-size:.64rem;letter-spacing:.06em}@media(max-width:640px){.bottom-line{margin-top:1.4rem}.bottom-footer{flex-direction:column;gap:.25rem}}</style>', unsafe_allow_html=True)
main()