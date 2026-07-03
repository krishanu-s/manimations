/**
 * NumberLineSort.tsx — drag-to-reorder number lines.
 *
 * Three number lines, each holding a set of labeled data points, are
 * stacked vertically. The user drags rows up and down to reorder them
 * (e.g. to answer a prompt like "order the sets from least to greatest
 * spread"). Row identity (color, name, points) travels with the row —
 * only its vertical position changes.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ─── Palette (categorical, fixed order — see dataviz color-formula) ──────────

const SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100"] as const;

const INK = {
  primary: "#0b0b0b",
  secondary: "#52514e",
  muted: "#898781",
  hairline: "#e1e0d9",
  baseline: "#c3c2b7",
  surface: "#fcfcfb",
  border: "rgba(11,11,11,0.10)",
  good: "#0ca30c",
  critical: "#d03b3b",
} as const;

// ─── Data ──────────────────────────────────────────────────────────────────

export interface DataPoint {
  label: string;
  value: number;
}

export interface NumberLineSet {
  id: string;
  name: string;
  color: string;
  points: DataPoint[];
}

const DEFAULT_DOMAIN: [number, number] = [0, 10];

const DEFAULT_SETS: NumberLineSet[] = [
  {
    id: "b",
    name: "Set B",
    color: SERIES_COLORS[1],
    points: [
      { label: "A", value: 0.8 },
      { label: "B", value: 2.5 },
      { label: "C", value: 5.0 },
      { label: "D", value: 7.6 },
      { label: "E", value: 9.3 },
    ],
  },
  {
    id: "a",
    name: "Set A",
    color: SERIES_COLORS[0],
    points: [
      { label: "A", value: 4.2 },
      { label: "B", value: 4.6 },
      { label: "C", value: 5.0 },
      { label: "D", value: 5.4 },
      { label: "E", value: 5.8 },
    ],
  },
  {
    id: "c",
    name: "Set C",
    color: SERIES_COLORS[2],
    points: [
      { label: "A", value: 2.8 },
      { label: "B", value: 3.9 },
      { label: "C", value: 5.0 },
      { label: "D", value: 6.1 },
      { label: "E", value: 7.3 },
    ],
  },
];

function stdDev(points: readonly DataPoint[]): number {
  const mean = points.reduce((s, p) => s + p.value, 0) / points.length;
  const variance =
    points.reduce((s, p) => s + (p.value - mean) ** 2, 0) / points.length;
  return Math.sqrt(variance);
}

// ─── Layout constants ─────────────────────────────────────────────────────

const ROW_HEIGHT = 138;
const CARD_HEIGHT = 118;
const VB_W = 640;
const VB_H = 64;
const PAD_X = 26;

function scaleX(value: number, domain: [number, number]): number {
  const [lo, hi] = domain;
  const t = (value - lo) / (hi - lo);
  return PAD_X + t * (VB_W - 2 * PAD_X);
}

// ─── Row (single number line) ─────────────────────────────────────────────

function NumberLineRow({ set, domain }: { set: NumberLineSet; domain: [number, number] }) {
  const axisY = VB_H / 2 + 4;
  const [lo, hi] = domain;
  const ticks = useMemo(() => {
    const arr: number[] = [];
    for (let v = Math.ceil(lo); v <= hi; v++) arr.push(v);
    return arr;
  }, [lo, hi]);

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      width="100%"
      height={VB_H}
      style={{ display: "block", overflow: "visible" }}
    >
      {ticks.map((t) => (
        <line
          key={t}
          x1={scaleX(t, domain)}
          x2={scaleX(t, domain)}
          y1={axisY - 4}
          y2={axisY + 4}
          stroke={INK.hairline}
          strokeWidth={1}
        />
      ))}
      <line
        x1={scaleX(lo, domain)}
        x2={scaleX(hi, domain)}
        y1={axisY}
        y2={axisY}
        stroke={INK.baseline}
        strokeWidth={1.5}
        strokeLinecap="round"
      />
      <text
        x={scaleX(lo, domain)}
        y={axisY + 20}
        fontSize={10}
        fill={INK.muted}
        textAnchor="middle"
        fontFamily="system-ui, sans-serif"
      >
        {lo}
      </text>
      <text
        x={scaleX(hi, domain)}
        y={axisY + 20}
        fontSize={10}
        fill={INK.muted}
        textAnchor="middle"
        fontFamily="system-ui, sans-serif"
      >
        {hi}
      </text>
      {set.points.map((p, i) => {
        const x = scaleX(p.value, domain);
        const above = i % 2 === 0;
        return (
          <g key={`${set.id}-${p.label}`}>
            <circle
              cx={x}
              cy={axisY}
              r={6}
              fill={set.color}
              stroke="#fff"
              strokeWidth={1.5}
            />
            <text
              x={x}
              y={above ? axisY - 14 : axisY + 24}
              fontSize={12}
              fontWeight={600}
              fill={INK.primary}
              textAnchor="middle"
              fontFamily="system-ui, sans-serif"
            >
              {p.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Main component ────────────────────────────────────────────────────────

export interface NumberLineSortProps {
  prompt?: string;
  domain?: [number, number];
  sets?: NumberLineSet[];
}

export default function NumberLineSort({
  prompt = "Order the sets from least to greatest spread.",
  domain = DEFAULT_DOMAIN,
  sets = DEFAULT_SETS,
}: NumberLineSortProps) {
  const [order, setOrder] = useState<string[]>(() => sets.map((s) => s.id));
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState(0);
  const [feedback, setFeedback] = useState<"correct" | "incorrect" | null>(null);

  const dragStartY = useRef(0);
  const dragStartOrder = useRef<string[]>([]);

  const setsById = useMemo(() => {
    const m = new Map<string, NumberLineSet>();
    for (const s of sets) m.set(s.id, s);
    return m;
  }, [sets]);

  const correctOrder = useMemo(
    () => [...sets].sort((s1, s2) => stdDev(s1.points) - stdDev(s2.points)).map((s) => s.id),
    [sets],
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>, id: string) => {
      dragStartY.current = e.clientY;
      dragStartOrder.current = order;
      setDraggingId(id);
      setDragOffset(0);
      setFeedback(null);
    },
    [order],
  );

  // Window-level listeners (rather than pointer capture on the row) because
  // the row's DOM node is reordered among siblings mid-drag as `order`
  // changes, which drops native pointer capture in some browsers.
  useEffect(() => {
    if (!draggingId) return;

    const handleMove = (e: PointerEvent) => {
      const deltaY = e.clientY - dragStartY.current;
      setDragOffset(deltaY);

      const startOrder = dragStartOrder.current;
      const startIndex = startOrder.indexOf(draggingId);
      const withoutDragged = startOrder.filter((x) => x !== draggingId);
      const rawIndex = startIndex + deltaY / ROW_HEIGHT;
      const newIndex = Math.min(Math.max(Math.round(rawIndex), 0), withoutDragged.length);
      const newOrder = [
        ...withoutDragged.slice(0, newIndex),
        draggingId,
        ...withoutDragged.slice(newIndex),
      ];
      setOrder(newOrder);
    };

    const handleUp = () => {
      setDraggingId(null);
      setDragOffset(0);
    };

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
    window.addEventListener("pointercancel", handleUp);
    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
      window.removeEventListener("pointercancel", handleUp);
    };
  }, [draggingId]);

  const handleCheck = useCallback(() => {
    setFeedback(order.join(",") === correctOrder.join(",") ? "correct" : "incorrect");
  }, [order, correctOrder]);

  return (
    <div
      style={{
        display: "inline-flex",
        flexDirection: "column",
        gap: "18px",
        fontFamily: "system-ui, sans-serif",
        color: INK.primary,
        padding: "20px",
        background: INK.surface,
        border: `1px solid ${INK.border}`,
        borderRadius: "10px",
        width: "min(680px, 100%)",
        boxSizing: "border-box",
      }}
    >
      <div style={{ fontSize: "16px", fontWeight: 600, color: INK.primary }}>{prompt}</div>

      <div style={{ position: "relative", height: order.length * ROW_HEIGHT - (ROW_HEIGHT - CARD_HEIGHT) }}>
        {order.map((id, index) => {
          const set = setsById.get(id);
          if (!set) return null;
          const isDragging = draggingId === id;
          const y = index * ROW_HEIGHT + (isDragging ? dragOffset : 0);
          return (
            <div
              key={id}
              onPointerDown={(e) => handlePointerDown(e, id)}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: CARD_HEIGHT,
                transform: `translateY(${y}px)`,
                transition: isDragging ? "none" : "transform 220ms ease",
                zIndex: isDragging ? 10 : 1,
                display: "flex",
                alignItems: "center",
                gap: "12px",
                padding: "0 16px",
                background: "#fff",
                border: `1px solid ${INK.border}`,
                borderRadius: "8px",
                boxShadow: isDragging
                  ? "0 8px 20px rgba(0,0,0,0.15)"
                  : "0 1px 2px rgba(0,0,0,0.04)",
                cursor: isDragging ? "grabbing" : "grab",
                touchAction: "none",
                userSelect: "none",
                boxSizing: "border-box",
              }}
            >
              <span style={{ color: INK.muted, fontSize: "14px", letterSpacing: "1px" }}>
                ⠿
              </span>
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  fontSize: "13px",
                  fontWeight: 600,
                  color: INK.secondary,
                  flexShrink: 0,
                  width: "56px",
                }}
              >
                <span
                  style={{
                    width: "10px",
                    height: "10px",
                    borderRadius: "50%",
                    background: set.color,
                    flexShrink: 0,
                  }}
                />
                {set.name}
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <NumberLineRow set={set} domain={domain} />
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
        <button
          onClick={handleCheck}
          style={{
            padding: "8px 16px",
            border: `1px solid ${INK.border}`,
            borderRadius: "6px",
            background: INK.surface,
            cursor: "pointer",
            fontSize: "13px",
            fontWeight: 600,
            color: INK.primary,
          }}
        >
          Check order
        </button>
        {feedback === "correct" && (
          <span style={{ color: INK.good, fontSize: "13px", fontWeight: 600 }}>
            ✓ Correct!
          </span>
        )}
        {feedback === "incorrect" && (
          <span style={{ color: INK.critical, fontSize: "13px", fontWeight: 600 }}>
            ✗ Not quite — keep adjusting.
          </span>
        )}
      </div>
    </div>
  );
}
