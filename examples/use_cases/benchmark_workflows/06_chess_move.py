# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Let the game generate legal moves, ask Wyat to rank them, then apply one."""

import json

import wyat
from examples.use_cases.common import load_model

FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def choose_move(model: wyat.Model, fen: str) -> dict[str, object]:
    try:
        import chess
    except ImportError as error:
        raise SystemExit(
            "This example needs python-chess. Run it with: uv run --with "
            "python-chess python -m "
            "examples.use_cases.benchmark_workflows.06_chess_move "
            "--model teilomillet/wyat-1-800m"
        ) from error

    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over():
        raise ValueError("supply a valid position with a move to play")
    # UCI describes the move without SAN's '+'/'#' hints about check or mate.
    candidates = {move.uci(): move.uci() for move in board.legal_moves}
    decision = model.decide(
        wyat.Request.choice(
            "Choose the strongest legal move for the side to move.",
            candidates,
            context={"fen": fen},
        )
    )
    # The game owns legality and state transitions; the model only selects.
    move = chess.Move.from_uci(decision.selected)
    if move not in board.legal_moves:
        raise ValueError("the selected move is no longer legal")
    notation = board.san(move)
    board.push(move)
    return {
        "move": decision.selected,
        "notation": notation,
        "next_fen": board.fen(),
        "checkmate": board.is_checkmate(),
        "decision": decision.to_dict(),
    }


def run(model: wyat.Model) -> dict[str, object]:
    return choose_move(model, FEN)


if __name__ == "__main__":
    print(json.dumps(run(load_model(__doc__)), ensure_ascii=False, indent=2))
