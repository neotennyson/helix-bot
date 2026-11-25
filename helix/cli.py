import argparse
import json
import statistics
import time as time_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backtest import BacktestEngine
from .broker_data import HistoricalDataError, fetch_dhan_daily_history, fetch_dhan_history, fetch_groww_history
from .config import BacktestConfig, load_config
from .data import download_yfinance_proxy, generate_demo_bars, load_bars, save_bars
from .optimize import WalkForwardOptimizer
from .paper import PaperTradingEngine
from .options import OptionConfig, run_option_research
from .research import run_research
from .universe import aggregate_portfolio, load_universe, rank_rows, scan_symbol


def main() -> None:
    parser = argparse.ArgumentParser(prog="helix")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo-data")
    demo.add_argument("--output", default="data/demo_15m.csv")
    demo.add_argument("--days", type=int, default=90)
    demo.add_argument("--symbol", default="NIFTY")
    proxy = subparsers.add_parser("download-spot-proxy")
    proxy.add_argument("--ticker", default="^NSEI")
    proxy.add_argument("--output", default="data/spot_proxy_15m.csv")
    proxy.add_argument("--symbol", default="NIFTY_PROXY")
    proxy.add_argument("--period", default="60d")
    proxy.add_argument("--interval", default="15m")
    groww = subparsers.add_parser("fetch-groww-history")
    groww.add_argument("--symbol", required=True)
    groww.add_argument("--start", required=True)
    groww.add_argument("--end", required=True)
    groww.add_argument("--interval", type=int, default=15)
    groww.add_argument("--exchange", default="NSE")
    groww.add_argument("--segment", default="FNO")
    groww.add_argument("--expiry")
    groww.add_argument("--output", default="data/groww_history.csv")
    dhan = subparsers.add_parser("fetch-dhan-history")
    dhan.add_argument("--security-id", required=True)
    dhan.add_argument("--start", required=True)
    dhan.add_argument("--end", required=True)
    dhan.add_argument("--interval", type=int, default=15)
    dhan.add_argument("--exchange-segment", default="NSE_FNO")
    dhan.add_argument("--instrument", default="FUTURE")
    dhan.add_argument("--expiry")
    dhan.add_argument("--no-oi", action="store_false", dest="include_oi")
    dhan.add_argument("--output", default="data/dhan_history.csv")
    dhan_daily = subparsers.add_parser("fetch-dhan-daily")
    dhan_daily.add_argument("--security-id", required=True)
    dhan_daily.add_argument("--start", required=True)
    dhan_daily.add_argument("--end", required=True)
    dhan_daily.add_argument("--exchange-segment", default="NSE_FNO")
    dhan_daily.add_argument("--instrument", default="FUTIDX")
    dhan_daily.add_argument("--expiry")
    dhan_daily.add_argument("--no-oi", action="store_false", dest="include_oi")
    dhan_daily.add_argument("--output", default="data/dhan_daily_history.csv")
    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--data", required=True)
    backtest.add_argument("--config", default="config/default.json")
    backtest.add_argument("--symbol")
    backtest.add_argument("--data-source", default="unspecified")
    backtest.add_argument("--output", default="reports/backtest.json")
    optimize = subparsers.add_parser("optimize")
    optimize.add_argument("--data", required=True)
    optimize.add_argument("--config", default="config/default.json")
    optimize.add_argument("--symbol")
    optimize.add_argument("--data-source", default="unspecified")
    optimize.add_argument("--output", default="reports/optimization.json")
    research = subparsers.add_parser("research")
    research.add_argument("--data", required=True)
    research.add_argument("--config", default="config/default.json")
    research.add_argument("--symbol")
    research.add_argument("--data-source", default="unspecified")
    research.add_argument("--output", default="reports/research.json")
    paper = subparsers.add_parser("paper")
    paper.add_argument("--data", required=True)
    paper.add_argument("--config", default="config/default.json")
    paper.add_argument("--state", default="data/paper_state.json")
    paper.add_argument("--symbol")
    paper.add_argument("--replay", action="store_true")
    paper.add_argument("--once", action="store_true")
    paper.add_argument("--loop", action="store_true")
    paper.add_argument("--poll", type=int, default=15)
    universe = subparsers.add_parser("universe")
    universe.add_argument("--manifest", default="config/equity_universe.json")
    universe.add_argument("--data-dir", default="data/equity")
    universe.add_argument("--template", default="config/equity_daily_template.json")
    universe.add_argument("--modes", default="volatility_breakout,mean_reversion,trend_breakout")
    universe.add_argument("--profiles", default="")
    universe.add_argument("--config-overrides", default="{}")
    universe.add_argument("--shortlist", type=int, default=8)
    universe.add_argument("--min-trades", type=int, default=20)
    universe.add_argument("--min-sharpe", type=float, default=0.5)
    universe.add_argument("--data-source", default="dhan_daily_continuous_equity")
    universe.add_argument("--output", default="reports/universe_scan.json")
    portfolio = subparsers.add_parser("portfolio-research")
    portfolio.add_argument("--manifest", default="config/equity_universe_wide.json")
    portfolio.add_argument("--data-dir", default="data/equity")
    portfolio.add_argument("--template", default="config/equity_daily_template.json")
    portfolio.add_argument("--profiles", default="config/equity_profiles.json")
    portfolio.add_argument("--top-k", type=int, default=25)
    portfolio.add_argument("--folds", type=int, default=4)
    portfolio.add_argument("--data-source", default="dhan_daily_continuous_equity")
    portfolio.add_argument("--output", default="reports/portfolio_research.json")
    options = subparsers.add_parser("options-research")
    options.add_argument("--data", required=True)
    options.add_argument("--symbol", required=True)
    options.add_argument("--strategies", default="covered_call,cash_secured_put")
    options.add_argument("--iv", default="0.12,0.14,0.16")
    options.add_argument("--deltas", default="0.25,0.35")
    options.add_argument("--holding-days", default="10,21,42")
    options.add_argument("--capital", type=float, default=10_000_000.0)
    options.add_argument("--max-capital-at-risk", type=float, default=0.25)
    options.add_argument("--data-source", default="synthetic_black_scholes_on_dhan_daily")
    options.add_argument("--output", default="reports/options_research.json")
    args = parser.parse_args()
    if args.command == "demo-data":
        bars = generate_demo_bars(symbol=args.symbol, days=args.days)
        save_bars(args.output, bars)
        print(json.dumps({"output": args.output, "bars": len(bars), "symbol": args.symbol}))
        return
    if args.command == "download-spot-proxy":
        bars = download_yfinance_proxy(args.ticker, args.output, args.symbol, args.period, args.interval)
        print(json.dumps({"output": args.output, "bars": len(bars), "ticker": args.ticker, "research_only": True}))
        return
    if args.command == "fetch-groww-history":
        bars = fetch_groww_history(args.symbol, args.start, args.end, args.interval, args.exchange, args.segment, args.expiry, args.output)
        print(json.dumps({"output": args.output, "bars": len(bars), "broker": "groww", "historical_only": True}))
        return
    if args.command == "fetch-dhan-history":
        bars = fetch_dhan_history(args.security_id, args.start, args.end, args.interval, args.exchange_segment, args.instrument, args.include_oi, args.expiry, args.output)
        print(json.dumps({"output": args.output, "bars": len(bars), "broker": "dhan", "historical_only": True}))
        return
    if args.command == "fetch-dhan-daily":
        bars = fetch_dhan_daily_history(args.security_id, args.start, args.end, args.exchange_segment, args.instrument, args.include_oi, args.expiry, args.output)
        print(json.dumps({"output": args.output, "bars": len(bars), "broker": "dhan", "historical_only": True}))
        return
    if args.command == "universe":
        run_universe(args)
        return
    if args.command == "portfolio-research":
        run_portfolio_research(args)
        return
    if args.command == "options-research":
        run_options_research(args)
        return
    config = load_config(args.config)
    if getattr(args, "symbol", None):
        config.symbol = args.symbol
    bars = load_bars(args.data, config.symbol)
    if args.command == "backtest":
        result = BacktestEngine(config).run(bars)
        payload = result.to_dict()
        payload["data_source"] = args.data_source
        payload["research_only"] = True
        write_json(args.output, payload)
        print(json.dumps({"output": args.output, "data_source": args.data_source, "final_equity": result.final_equity, "metrics": result.metrics}, default=str))
        return
    if args.command == "optimize":
        report = WalkForwardOptimizer(config).run(bars)
        report["data_source"] = args.data_source
        report["research_only"] = True
        write_json(args.output, report)
        print(json.dumps({"output": args.output, "data_source": args.data_source, "promotion": report["promotion"], "selected_parameters": report["selected_parameters"]}, default=str))
        return
    if args.command == "research":
        report = run_research(config, bars, args.data_source)
        write_json(args.output, report)
        print(json.dumps({"output": args.output, "quality": report["quality"], "promotion": report["validation"]["promotion"], "paper_review_candidate": report["paper_review_candidate"]}, default=str))
        return
    if args.command == "paper":
        run_paper(args, config, bars)
        return
    raise RuntimeError("unsupported command")


def run_universe(args: argparse.Namespace) -> None:
    members = load_universe(args.manifest, args.data_dir)
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    base_overrides = json.loads(args.config_overrides)
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8")) if args.profiles else {}
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    jobs: list[tuple[str, str, dict[str, Any]]] = []
    for mode in modes:
        mode_profiles = profiles.get(mode) or [{}]
        if isinstance(mode_profiles, dict):
            mode_profiles = [{"label": key, **value} for key, value in mode_profiles.items()]
        for profile in mode_profiles:
            label = str(profile.get("label", "base"))
            overrides = {key: value for key, value in profile.items() if key != "label"}
            jobs.append((mode, label, overrides))
    scans: list[dict[str, Any]] = []
    for member in members:
        symbol_scans: list[dict[str, Any]] = []
        for mode, label, overrides in jobs:
            merged = json.loads(json.dumps(base_overrides))
            for key, value in overrides.items():
                if key in {"strategy", "risk", "costs"} and isinstance(value, dict):
                    merged.setdefault(key, {}).update(value)
                else:
                    merged[key] = value
            try:
                scan = scan_symbol(template, member, mode, merged)
                scan["profile"] = label
            except (ValueError, HistoricalDataError) as error:
                scan = {
                    "symbol": member.symbol,
                    "strategy_mode": mode,
                    "profile": label,
                    "error": str(error),
                    "metrics": {},
                    "initial_capital": 0.0,
                    "equity_curve": [],
                    "trades": [],
                }
            symbol_scans.append(scan)
            scans.append(scan)
        best = max(symbol_scans, key=lambda item: float(item["metrics"].get("sharpe", -99.0)))
        metrics = best["metrics"]
        print(
            f"{member.symbol:12s} best={best['strategy_mode']:22s}/{best['profile']:14s} "
            f"ret {float(metrics.get('total_return', 0.0)) * 100:8.2f}%  "
            f"sharpe {float(metrics.get('sharpe', 0.0)):6.3f}  "
            f"trades {int(metrics.get('trades', 0) or 0):4d}",
            flush=True,
        )
    rows = rank_rows(scans, min_trades=args.min_trades, min_sharpe=args.min_sharpe)
    profile_stats = summarize_profiles(rows)
    benchmark_returns = [float(row["benchmark_return"]) for row in rows if row.get("benchmark_return") is not None]
    buy_hold_reference = {
        "label": "equal_weight_buy_and_hold_of_scanned_symbols",
        "note": "beta reference over each symbol's first-to-last close, not a traded strategy; no costs, no timing",
        "mean_total_return": statistics.mean(benchmark_returns) if benchmark_returns else None,
        "median_total_return": statistics.median(benchmark_returns) if benchmark_returns else None,
        "share_of_symbols_above_strategy_median": None,
    }
    by_symbol_mode: dict[str, dict[str, Any]] = {}
    for scan in scans:
        key = f"{scan['symbol']}:{scan['strategy_mode']}:{scan.get('profile', 'base')}"
        by_symbol_mode[key] = {
            "symbol": scan["symbol"],
            "strategy_mode": scan["strategy_mode"],
            "profile": scan.get("profile", "base"),
            "error": scan.get("error"),
            "metrics": scan.get("metrics", {}),
        }
    shortlist = rows[: max(1, args.shortlist)]
    sleeves = []
    for row in shortlist:
        key = f"{row['symbol']}:{row['strategy_mode']}:{row.get('profile', 'base')}"
        sleeves.extend([scan for scan in scans if f"{scan['symbol']}:{scan['strategy_mode']}:{scan.get('profile', 'base')}" == key])
    portfolio = aggregate_portfolio(sleeves, int(template.get("bar_minutes", 1440)))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": args.data_source,
        "research_only": True,
        "screen_is_in_sample": True,
        "universe_size": len(members),
        "modes": modes,
        "template": template,
        "config_overrides": overrides,
        "promotion_rules": {
            "note": "screening only; the walk-forward holdout gate is the decision of record",
            "minimum_trades": args.min_trades,
            "minimum_sharpe": args.min_sharpe,
            "cash_hurdle_annual": float((template.get("risk") or {}).get("cash_hurdle_annual", 0.06)),
            "minimum_cagr": 0.06,
            "minimum_profit_factor": 1.10,
            "minimum_max_drawdown": -0.15,
        },
        "ranking": rows,
        "profile_summary": profile_stats,
        "buy_and_hold_reference": buy_hold_reference,
        "results": by_symbol_mode,
        "shortlist": [
            {
                "symbol": row["symbol"],
                "strategy_mode": row["strategy_mode"],
                "profile": row.get("profile", "base"),
                "cagr": row["cagr"],
                "sharpe": row["sharpe"],
                "trades": row["trades"],
            }
            for row in shortlist
        ],
        "portfolio": {
            "status": portfolio.get("status"),
            "weighting": portfolio.get("weighting"),
            "sleeves": [row["symbol"] for row in shortlist],
            "metrics": portfolio.get("metrics", {}),
            "average_pairwise_correlation": (portfolio.get("metrics") or {}).get("average_pairwise_correlation"),
        },
        "qualifying_symbols": [row["symbol"] for row in rows if row["qualifies"]],
    }
    write_json(args.output, report)
    metrics = portfolio.get("metrics", {})
    print(
        json.dumps(
            {
                "output": args.output,
                "universe_size": len(members),
                "scans": len(scans),
                "qualifying_symbols": len(report["qualifying_symbols"]),
                "portfolio_cagr": metrics.get("cagr"),
                "portfolio_sharpe": metrics.get("sharpe"),
                "portfolio_max_drawdown": metrics.get("max_drawdown"),
                "portfolio_trades": metrics.get("trades"),
            },
            default=str,
        )
    )


def summarize_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["strategy_mode"], row.get("profile", "base")), []).append(row)
    summary: list[dict[str, Any]] = []
    for (mode, profile), items in grouped.items():
        returns = [float(item["total_return"] or 0.0) for item in items]
        sharpes = [float(item["sharpe"] or 0.0) for item in items]
        summary.append(
            {
                "strategy_mode": mode,
                "profile": profile,
                "symbols": len(items),
                "median_total_return": statistics.median(returns),
                "mean_total_return": statistics.mean(returns),
                "median_sharpe": statistics.median(sharpes),
                "mean_sharpe": statistics.mean(sharpes),
                "best_sharpe": max(sharpes),
                "median_trades": statistics.median([int(item["trades"] or 0) for item in items]),
                "symbols_in_profit": sum(1 for value in returns if value > 0.0),
            }
        )
    summary.sort(key=lambda item: item["median_sharpe"], reverse=True)
    return summary


def run_portfolio_research(args: argparse.Namespace) -> None:
    from .portfolio_research import run_portfolio_walk_forward

    members = load_universe(args.manifest, args.data_dir)
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    modes = {
        mode: [profile for profile in (value if isinstance(value, list) else list(value.values())) if isinstance(profile, dict)]
        for mode, value in profiles.items()
    }
    bars_by_symbol = {member.symbol: load_bars(member.path, member.security_id) for member in members}
    print(f"universe {len(members)} symbols | top_k {args.top_k} | folds {args.folds}", flush=True)
    report = run_portfolio_walk_forward(
        template, members, modes, bars_by_symbol, top_k=args.top_k, folds=args.folds,
        bar_minutes=int(template.get("bar_minutes", 1440)),
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["data_source"] = args.data_source
    report["research_only"] = True
    write_json(args.output, report)
    print(json.dumps({
        "output": args.output,
        "status": report.get("status"),
        "stitched_metrics": report.get("stitched_metrics"),
        "promotion": report.get("promotion", {}).get("passed"),
    }, default=str))


def run_options_research(args: argparse.Namespace) -> None:
    bars = load_bars(args.data, args.symbol)
    runs: list[dict[str, Any]] = []
    for strategy in [item.strip() for item in args.strategies.split(",") if item.strip()]:
        for implied_volatility in [float(item) for item in args.iv.split(",") if item.strip()]:
            for delta_target in [float(item) for item in args.deltas.split(",") if item.strip()]:
                for holding_days in [int(item) for item in args.holding_days.split(",") if item.strip()]:
                    config = OptionConfig(
                        underlying_symbol=args.symbol,
                        strategy=strategy,
                        implied_volatility=implied_volatility,
                        delta_target=delta_target,
                        holding_days=holding_days,
                        initial_capital=args.capital,
                        max_capital_at_risk_pct=args.max_capital_at_risk,
                    )
                    report = run_option_research(config, bars)
                    report["grid"] = {
                        "strategy": strategy,
                        "implied_volatility": implied_volatility,
                        "delta_target": delta_target,
                        "holding_days": holding_days,
                    }
                    runs.append(report)
                    metrics = report.get("metrics", {})
                    print(
                        f"{strategy:18s} iv={implied_volatility:.2f} d={delta_target:.2f} h={holding_days:2d} "
                        f"cagr {float(metrics.get('cagr', 0.0)) * 100:6.2f}% sharpe {float(metrics.get('sharpe', 0.0)):6.3f} "
                        f"dd {float(metrics.get('max_drawdown', 0.0)) * 100:6.2f}% trades {int(metrics.get('trades', 0) or 0):4d}"
                    )
    ranked = sorted(
        runs,
        key=lambda item: float(item.get("metrics", {}).get("sharpe", -99.0)),
        reverse=True,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "real_option_quotes": False,
        "data_source": args.data_source,
        "underlying_symbol": args.symbol,
        "underlying_bars": len(bars),
        "disclaimer": (
            "Black-Scholes premiums on a real underlying path with an assumed constant IV surface. "
            "No bid/ask, no skew, no IV term structure, no vega shock. Modelled research only."
        ),
        "runs": len(ranked),
        "best_by_sharpe": ranked[0].get("grid") if ranked else None,
        "best_sharpe": float(ranked[0].get("metrics", {}).get("sharpe", 0.0)) if ranked else None,
        "any_passed_promotion": any(bool(item.get("promotion", {}).get("passed")) for item in runs),
        "results": [
            {
                "grid": item["grid"],
                "metrics": {
                    key: item["metrics"].get(key)
                    for key in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "profit_factor", "trades", "win_rate")
                },
                "exit_reason_counts": item.get("exit_reason_counts"),
                "excess_cagr_over_cash": item["metrics"].get("excess_cagr_over_cash"),
            }
            for item in ranked
        ],
    }
    write_json(args.output, payload)
    print(json.dumps({"output": args.output, "runs": len(ranked), "best_sharpe": payload["best_sharpe"], "any_passed_promotion": payload["any_passed_promotion"]}, default=str))


def run_paper(args: argparse.Namespace, config: BacktestConfig, bars: list) -> None:
    engine = PaperTradingEngine(config, args.state)
    if args.replay:
        start = 0
    else:
        engine.initialize_history(bars)
        last_timestamp = engine.state.get("last_timestamp")
        start = 0
        if last_timestamp is not None:
            start = next((index for index, bar in enumerate(bars) if bar.timestamp.isoformat() > last_timestamp), len(bars))
    if args.loop:
        while True:
            bars = load_bars(args.data, config.symbol)
            start_timestamp = engine.state.get("last_timestamp")
            for bar in bars:
                if start_timestamp is None or bar.timestamp.isoformat() > start_timestamp:
                    engine.process_bar(bar)
            time_module.sleep(max(1, args.poll))
    else:
        if start >= len(bars):
            print(json.dumps(paper_summary(engine.state), default=str))
            return
        for bar in bars[start:]:
            engine.process_bar(bar)
        print(json.dumps(paper_summary(engine.state), default=str))


def paper_summary(state: dict) -> dict:
    return {
        "mode": state.get("mode"),
        "broker": state.get("broker"),
        "symbol": state.get("symbol"),
        "equity": state.get("equity"),
        "realized_equity": state.get("realized_equity"),
        "unrealized_pnl": state.get("unrealized_pnl"),
        "equity_includes_unrealized": state.get("equity_includes_unrealized"),
        "cash": state.get("cash"),
        "status": state.get("status"),
        "trades": len(state.get("trades", [])),
        "last_timestamp": state.get("last_timestamp"),
        "position": state.get("position"),
        "pending": state.get("pending"),
    }


def write_json(path: str, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=str)


if __name__ == "__main__":
    main()
