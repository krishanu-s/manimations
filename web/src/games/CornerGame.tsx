/**
 * CornerGame.tsx — "don't land in the corner" chessboard game.
 *
 * A single piece sits on an n×n board. Two players alternate moving it,
 * always toward the bottom-left (rook: left/down, queen: left/down/diagonal,
 * king: one step left/down/diagonal). Whoever moves the piece onto the
 * bottom-left square loses.
 *
 * Left: the playable board (pick a start square, then make moves).
 * Right: a notes board of the same size where the user paints squares as
 * "first player wins" / "second player wins" while working out the pattern.
 */

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

// ─── Palette ────────────────────────────────────────────────────────────────

const INK = {
  primary: "#0b0b0b",
  secondary: "#52514e",
  muted: "#898781",
  surface: "#fcfcfb",
  border: "rgba(11,11,11,0.10)",
  critical: "#d03b3b",
} as const;

const SQUARE = { light: "#f0efe9", dark: "#cfccbf" } as const;

const PLAYER_COLORS = { 1: "#2a78d6", 2: "#eda100" } as const;

const MAX_BOARD_PX = 320;
// Below this board width the two panels stack instead of sitting side by side.
const MIN_BOARD_PX = 150;
const LABEL_PX = 18;
const PANEL_GAP = 16;
const PANEL_PAD = 16;
// Horizontal space a panel needs beyond its board: padding, panel border,
// board border, and the rank-label column.
const PANEL_CHROME = 2 * PANEL_PAD + 2 + 2 + LABEL_PX;

// ─── Game rules ─────────────────────────────────────────────────────────────

type Piece = "rook" | "queen" | "king";
type Player = 1 | 2;
/** 0 = unmarked, 1 = first player wins, 2 = second player wins. */
type Mark = 0 | Player;

interface Sq {
  x: number; // file, 0 = leftmost
  y: number; // rank, 0 = bottom
}

const PIECES: Record<Piece, { label: string; glyph: string; rule: string }> = {
  rook: {
    label: "Rook",
    glyph: "♜",
    rule: "Moves any number of squares left or down.",
  },
  queen: {
    label: "Queen",
    glyph: "♛",
    rule: "Moves any number of squares left, down, or diagonally down-left.",
  },
  king: {
    label: "King",
    glyph: "♚",
    rule: "Moves one square left, down, or diagonally down-left.",
  },
};

const PIECE_ORDER: Piece[] = ["rook", "queen", "king"];
const SIZE_OPTIONS = Array.from({ length: 11 }, (_, i) => i + 2); // 2..12
const DEFAULT_SIZE = 8;

const isCorner = (s: Sq) => s.x === 0 && s.y === 0;
const sqKey = (s: Sq) => `${s.x},${s.y}`;
const sqName = (s: Sq) => `${String.fromCharCode(97 + s.x)}${s.y + 1}`;

function legalMoves(piece: Piece, { x, y }: Sq): Sq[] {
  const moves: Sq[] = [];
  const reach = piece === "king" ? 1 : Infinity;
  for (let k = 1; k <= Math.min(x, reach); k++) moves.push({ x: x - k, y });
  for (let k = 1; k <= Math.min(y, reach); k++) moves.push({ x, y: y - k });
  if (piece !== "rook") {
    for (let k = 1; k <= Math.min(x, y, reach); k++) {
      moves.push({ x: x - k, y: y - k });
    }
  }
  return moves;
}

// ─── Board frame ────────────────────────────────────────────────────────────

interface BoardFrameProps {
  size: number;
  boardPx: number;
  renderSquare: (s: Sq, base: React.CSSProperties) => React.ReactNode;
  gridProps?: React.HTMLAttributes<HTMLDivElement> & {
    ref?: React.Ref<HTMLDivElement>;
  };
}

/** Square grid with rank labels on the left and file labels underneath. */
function BoardFrame({ size, boardPx, renderSquare, gridProps }: BoardFrameProps) {
  const cell = boardPx / size;
  const labelStyle: React.CSSProperties = {
    fontSize: "11px",
    color: INK.muted,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  };

  const squares: React.ReactNode[] = [];
  for (let row = 0; row < size; row++) {
    const y = size - 1 - row;
    for (let x = 0; x < size; x++) {
      const base: React.CSSProperties = {
        width: cell,
        height: cell,
        background: (x + y) % 2 === 0 ? SQUARE.dark : SQUARE.light,
        position: "relative",
        boxSizing: "border-box",
      };
      squares.push(renderSquare({ x, y }, base));
    }
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `${LABEL_PX}px ${boardPx}px`,
        gridTemplateRows: `${boardPx}px ${LABEL_PX}px`,
      }}
    >
      <div style={{ display: "grid", gridTemplateRows: `repeat(${size}, 1fr)` }}>
        {Array.from({ length: size }, (_, row) => (
          <div key={row} style={labelStyle}>
            {size - row}
          </div>
        ))}
      </div>
      <div
        {...gridProps}
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${size}, ${cell}px)`,
          gridTemplateRows: `repeat(${size}, ${cell}px)`,
          border: `1px solid ${INK.border}`,
          boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
          touchAction: "none",
          userSelect: "none",
          ...gridProps?.style,
        }}
      >
        {squares}
      </div>
      <div />
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${size}, 1fr)` }}>
        {Array.from({ length: size }, (_, x) => (
          <div key={x} style={labelStyle}>
            {String.fromCharCode(97 + x)}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Small UI pieces ────────────────────────────────────────────────────────

function Panel({
  title,
  boardPx,
  children,
}: {
  title: string;
  boardPx: number;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        padding: PANEL_PAD,
        background: INK.surface,
        border: `1px solid ${INK.border}`,
        borderRadius: "10px",
        width: boardPx + PANEL_CHROME,
        boxSizing: "border-box",
      }}
    >
      <div style={{ fontSize: "15px", fontWeight: 600 }}>{title}</div>
      {children}
    </div>
  );
}

function buttonStyle(active = false, disabled = false): React.CSSProperties {
  return {
    padding: "6px 12px",
    border: `1px solid ${active ? INK.primary : INK.border}`,
    borderRadius: "6px",
    background: active ? INK.primary : "#fff",
    color: active ? "#fff" : INK.primary,
    cursor: disabled ? "default" : "pointer",
    opacity: disabled ? 0.4 : 1,
    fontSize: "13px",
    fontWeight: 600,
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    whiteSpace: "nowrap",
  };
}

function Swatch({ color }: { color: string }) {
  return (
    <span
      style={{
        width: 12,
        height: 12,
        borderRadius: 3,
        background: color,
        border: `1px solid ${INK.border}`,
        display: "inline-block",
      }}
    />
  );
}

// ─── Main component ─────────────────────────────────────────────────────────

export default function CornerGame() {
  // ── Responsive sizing: fit both panels side by side in the container ──
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(
    2 * (MAX_BOARD_PX + PANEL_CHROME) + PANEL_GAP,
  );

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => setContainerWidth(el.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const sideBySidePx = Math.floor((containerWidth - PANEL_GAP) / 2 - PANEL_CHROME);
  const sideBySide = sideBySidePx >= MIN_BOARD_PX;
  const boardPx = Math.max(
    0,
    Math.min(MAX_BOARD_PX, sideBySide ? sideBySidePx : containerWidth - PANEL_CHROME),
  );

  const [piece, setPiece] = useState<Piece>("rook");
  const [size, setSize] = useState(DEFAULT_SIZE);
  const [start, setStart] = useState<Sq | null>(null);
  const [history, setHistory] = useState<Sq[]>([]);

  // Notes are kept per (piece, size) so switching pieces doesn't wipe work.
  const [notesByKey, setNotesByKey] = useState<Record<string, Mark[]>>({});
  const [brush, setBrush] = useState<Mark>(1);

  const notesKey = `${piece}-${size}`;
  const notes = notesByKey[notesKey] ?? new Array<Mark>(size * size).fill(0);

  // ── Game state ──
  const pos: Sq | null = history[history.length - 1] ?? start;
  const phase: "setup" | "playing" | "over" =
    pos === null ? "setup" : isCorner(pos) ? "over" : "playing";
  const toMove: Player = history.length % 2 === 0 ? 1 : 2;
  const loser: Player = toMove === 1 ? 2 : 1; // only meaningful when over

  const legal = useMemo(() => {
    if (phase !== "playing" || !pos) return new Set<string>();
    return new Set(legalMoves(piece, pos).map(sqKey));
  }, [phase, pos, piece]);

  const moveNumber = useMemo(() => {
    const m = new Map<string, number>();
    history.forEach((s, i) => m.set(sqKey(s), i + 1));
    return m;
  }, [history]);

  const newStart = useCallback(() => {
    setStart(null);
    setHistory([]);
  }, []);

  const handleGameSquare = (s: Sq) => {
    if (phase === "setup") {
      if (isCorner(s)) return;
      setStart(s);
      setHistory([]);
    } else if (phase === "playing" && legal.has(sqKey(s))) {
      setHistory((h) => [...h, s]);
    }
  };

  // ── Notes painting (click or drag) ──
  const notesGridRef = useRef<HTMLDivElement>(null);
  const strokeRef = useRef<Mark | null>(null);

  const cellFromPointer = (e: React.PointerEvent): Sq | null => {
    const rect = notesGridRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * size);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * size);
    if (col < 0 || col >= size || row < 0 || row >= size) return null;
    return { x: col, y: size - 1 - row };
  };

  const paint = (s: Sq, mark: Mark) => {
    if (isCorner(s)) return;
    const i = s.y * size + s.x;
    setNotesByKey((prev) => {
      const cur = prev[notesKey] ?? new Array<Mark>(size * size).fill(0);
      if (cur[i] === mark) return prev;
      const next = cur.slice();
      next[i] = mark;
      return { ...prev, [notesKey]: next };
    });
  };

  const onNotesPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const s = cellFromPointer(e);
    if (!s || isCorner(s)) return;
    // Clicking a square already in the brush color clears it instead.
    const existing = notes[s.y * size + s.x];
    strokeRef.current = brush !== 0 && existing === brush ? 0 : brush;
    e.currentTarget.setPointerCapture(e.pointerId);
    paint(s, strokeRef.current);
  };

  const onNotesPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (strokeRef.current === null) return;
    const s = cellFromPointer(e);
    if (s) paint(s, strokeRef.current);
  };

  const endStroke = () => {
    strokeRef.current = null;
  };

  useEffect(() => {
    window.addEventListener("pointerup", endStroke);
    return () => window.removeEventListener("pointerup", endStroke);
  }, []);

  const clearNotes = () =>
    setNotesByKey((prev) => {
      const { [notesKey]: _, ...rest } = prev;
      return rest;
    });

  // ── Status text ──
  let status: React.ReactNode;
  if (phase === "setup") {
    status = <>Click any square (other than a1) to place the {PIECES[piece].label.toLowerCase()}.</>;
  } else if (phase === "playing") {
    status = (
      <>
        <span style={{ color: PLAYER_COLORS[toMove], fontWeight: 700 }}>
          Player {toMove}
        </span>{" "}
        to move — click a highlighted square.
      </>
    );
  } else {
    const winner: Player = loser === 1 ? 2 : 1;
    status = (
      <>
        Player {loser} landed on a1 and loses.{" "}
        <span style={{ color: PLAYER_COLORS[winner], fontWeight: 700 }}>
          Player {winner} wins!
        </span>
      </>
    );
  }

  const glyphSize = (boardPx / size) * 0.7;

  return (
    <div
      ref={containerRef}
      style={{
        display: "flex",
        flexWrap: sideBySide ? "nowrap" : "wrap",
        width: "100%",
        gap: PANEL_GAP,
        alignItems: "flex-start",
        fontFamily: "system-ui, sans-serif",
        color: INK.primary,
      }}
    >
      {/* ── Game board ── */}
      <Panel title="Game" boardPx={boardPx}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center" }}>
          {PIECE_ORDER.map((p) => (
            <button
              key={p}
              style={buttonStyle(piece === p)}
              onClick={() => {
                setPiece(p);
                newStart();
              }}
            >
              <span style={{ fontSize: "15px" }}>{PIECES[p].glyph}</span>
              {PIECES[p].label}
            </button>
          ))}
          <label style={{ fontSize: "13px", marginLeft: "auto", display: "flex", gap: "6px", alignItems: "center" }}>
            Size
            <select
              value={size}
              onChange={(e) => {
                setSize(Number(e.target.value));
                newStart();
              }}
              style={{ fontSize: "13px", padding: "3px 4px" }}
            >
              {SIZE_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n}×{n}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div style={{ fontSize: "12px", color: INK.secondary }}>
          {PIECES[piece].label}: {PIECES[piece].rule} Whoever moves onto a1 loses.
        </div>

        <BoardFrame
          size={size}
          boardPx={boardPx}
          renderSquare={(s, base) => {
            const k = sqKey(s);
            const here = pos !== null && pos.x === s.x && pos.y === s.y;
            const isLegal = legal.has(k);
            const clickable = phase === "setup" ? !isCorner(s) : isLegal;
            const num = moveNumber.get(k);
            const isStart = start !== null && start.x === s.x && start.y === s.y;
            return (
              <div
                key={k}
                onClick={() => handleGameSquare(s)}
                title={isCorner(s) ? "a1 — moving here loses" : sqName(s)}
                style={{
                  ...base,
                  cursor: clickable ? "pointer" : "default",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: isStart && !here ? `inset 0 0 0 2px ${INK.muted}` : undefined,
                }}
              >
                {isCorner(s) && !here && (
                  <span style={{ color: INK.critical, opacity: 0.55, fontSize: glyphSize * 0.6 }}>✕</span>
                )}
                {num !== undefined && !here && (
                  <span
                    style={{
                      position: "absolute",
                      top: 1,
                      left: 3,
                      fontSize: "9px",
                      color: INK.secondary,
                    }}
                  >
                    {num}
                  </span>
                )}
                {isLegal && (
                  <span
                    style={{
                      position: "absolute",
                      width: "34%",
                      height: "34%",
                      borderRadius: "50%",
                      background: isCorner(s) ? "transparent" : "rgba(11,11,11,0.22)",
                      border: isCorner(s) ? `2px solid ${INK.critical}` : undefined,
                      boxSizing: "border-box",
                    }}
                  />
                )}
                {here && (
                  <span
                    style={{
                      fontSize: glyphSize,
                      lineHeight: 1,
                      color: phase === "over" ? INK.critical : INK.primary,
                    }}
                  >
                    {PIECES[piece].glyph}
                  </span>
                )}
              </div>
            );
          }}
        />

        <div style={{ fontSize: "13px", minHeight: "1.4em" }}>{status}</div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
          <button
            style={buttonStyle(false, history.length === 0)}
            disabled={history.length === 0}
            onClick={() => setHistory((h) => h.slice(0, -1))}
          >
            Undo
          </button>
          <button
            style={buttonStyle(false, history.length === 0)}
            disabled={history.length === 0}
            onClick={() => setHistory([])}
          >
            Restart from {start ? sqName(start) : "start"}
          </button>
          <button
            style={buttonStyle(false, phase === "setup")}
            disabled={phase === "setup"}
            onClick={newStart}
          >
            New starting square
          </button>
        </div>

        {start && history.length > 0 && (
          <div style={{ fontSize: "12px", color: INK.secondary, lineHeight: 1.6 }}>
            {history.map((s, i) => {
              const from = history[i - 1] ?? start;
              const player: Player = i % 2 === 0 ? 1 : 2;
              return (
                <span key={i} style={{ marginRight: "10px", whiteSpace: "nowrap" }}>
                  <span style={{ color: PLAYER_COLORS[player], fontWeight: 700 }}>
                    {i + 1}.
                  </span>{" "}
                  {sqName(from)}→{sqName(s)}
                </span>
              );
            })}
          </div>
        )}
      </Panel>

      {/* ── Notes board ── */}
      <Panel title="Notes" boardPx={boardPx}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
          <button style={buttonStyle(brush === 1)} onClick={() => setBrush(1)}>
            <Swatch color={PLAYER_COLORS[1]} /> First player wins
          </button>
          <button style={buttonStyle(brush === 2)} onClick={() => setBrush(2)}>
            <Swatch color={PLAYER_COLORS[2]} /> Second player wins
          </button>
          <button style={buttonStyle(brush === 0)} onClick={() => setBrush(0)}>
            Eraser
          </button>
        </div>

        <div style={{ fontSize: "12px", color: INK.secondary }}>
          Mark who wins from each starting square. Click or drag to paint.
        </div>

        <BoardFrame
          size={size}
          boardPx={boardPx}
          gridProps={{
            ref: notesGridRef,
            onPointerDown: onNotesPointerDown,
            onPointerMove: onNotesPointerMove,
            onPointerCancel: endStroke,
            style: { cursor: "crosshair" },
          }}
          renderSquare={(s, base) => {
            const mark = notes[s.y * size + s.x] ?? 0;
            const here = pos !== null && pos.x === s.x && pos.y === s.y;
            return (
              <div
                key={sqKey(s)}
                style={{
                  ...base,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {mark !== 0 && (
                  <span
                    style={{
                      position: "absolute",
                      inset: 0,
                      background: PLAYER_COLORS[mark],
                      opacity: 0.8,
                    }}
                  />
                )}
                {isCorner(s) && (
                  <span style={{ color: INK.critical, opacity: 0.55, fontSize: glyphSize * 0.6 }}>✕</span>
                )}
                {here && (
                  <span
                    style={{
                      position: "absolute",
                      inset: "18%",
                      borderRadius: "50%",
                      border: `2px dashed ${INK.primary}`,
                      opacity: 0.6,
                    }}
                  />
                )}
              </div>
            );
          }}
        />

        <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
          <button style={buttonStyle()} onClick={clearNotes}>
            Clear notes
          </button>
          <span style={{ fontSize: "11px", color: INK.muted }}>
            Click a square in the brush color to clear it. Notes are kept
            separately for each piece and board size.
          </span>
        </div>
      </Panel>
    </div>
  );
}
