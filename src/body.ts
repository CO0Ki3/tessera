/**
 * body.ts — a small expression IR for kernel bodies, and the walker that builds it.
 *
 * WHY THIS EXISTS. Before it, `checkCanonicalBody` recognised schedules by counting
 * operator names against a per-kernel template: "1 zeros, N loops, 1 mma, 1 store" for
 * matmul, "2 rowFill, 3 loops, 1 rowMax, 1 rowSum, 1 store" for softmax. Adding
 * layernorm produced exactly what docs/004 part 2 predicted it would —
 *
 *     the softmax schedule is 2 rowFill, 3 loops, 1 rowMax, 1 rowSum, 1 store;
 *     this body is 2 rowFill / 2 loops / 0 rowMax / 2 rowSum / 1 store
 *
 * — a third template. The pre-registered response was to generalise rather than to add
 * the branch, because a vocabulary that grows one hand-written case per kernel family is
 * a template collection however good its front end is.
 *
 * So the row-wise family is now READ rather than matched: accumulators, the passes over
 * the reduce axis, the derived row values and the store come out of the statements the
 * author wrote. softmax and layernorm differ in how many accumulators and passes there
 * are and what each does, and none of that is written down anywhere in the compiler.
 *
 * The vocabulary is still closed — every operator below is named here, and anything else
 * is refused rather than approximated. What changed is that the SHAPE of a body is no
 * longer enumerated.
 */

import ts from "typescript";
import type { PadName } from "./ir.ts";

/** A block-valued expression. */
export type Expr =
  /** `<binding>.tile(i, j)`, optionally `.pad(identity)`. */
  | { readonly k: "tile"; readonly binding: string; readonly pad?: PadName;
      readonly coords: readonly Coord[]; readonly transposed: boolean }
  | { readonly k: "unary"; readonly op: "sq" | "exp"; readonly a: Expr }
  /** A block combined with a row value, broadcast along the block's columns. */
  | { readonly k: "rowOp"; readonly op: "sub" | "mul" | "div"; readonly a: Expr; readonly row: RowExpr };

/** A row-valued expression: one value per row the workgroup owns. */
export type RowExpr =
  | { readonly k: "acc"; readonly name: string }
  /**
   * A row reduction of a computed FRAGMENT rather than of a staged tile.
   *
   * The first three families only ever reduced something read from a buffer, one
   * block per loop iteration, so a reduction was always an update inside a pass.
   * Reducing a fragment happens outside any loop: the fragment is already whole.
   *
   * `init` is the identity the accumulator was filled with, carried so the
   * emitter knows what an empty lane contributes — and so the compiler pass can
   * refuse `max` over a fragment whose reduced axis is ragged, where the
   * annihilating zero `mma` guarantees is the wrong identity.
   */
  | { readonly k: "reduceFrag"; readonly op: "max" | "sum";
      readonly a: FragExpr; readonly init: PadName }
  | { readonly k: "mean"; readonly a: RowExpr }
  | { readonly k: "rstd"; readonly sumSq: RowExpr; readonly mean: RowExpr; readonly eps: number };

/** How an accumulator is laid out per invocation. */
export type AccKind = "row" | "frag";

export type Update =
  /** Fold one operand cell into a row accumulator. */
  | { readonly k: "fold"; readonly acc: string; readonly op: "max" | "sum"; readonly value: Expr }
  /**
   * The same fold over a FRAGMENT the pass just computed, rather than over a tile
   * read from memory. The value is whole and register-resident, so this crosses
   * lanes where the tile form reads one block per step.
   */
  | { readonly k: "foldFrag"; readonly acc: string; readonly op: "max" | "sum";
      readonly value: FragExpr }
  /** Outer-product two operand slices into a 2-D fragment. */
  | { readonly k: "mma"; readonly acc: string; readonly a: Expr; readonly b: Expr }
  /**
   * A contraction whose first operand is a fragment the body already computed.
   *
   * Attention's `O = P·V`. The fragment is laid out by the contraction that built
   * it, and this one sums along one of THAT contraction's accumulate axes, so
   * each invocation holds only its own slice of what is being summed. Which is
   * why the emitter has to stage it rather than read it in place.
   */
  | { readonly k: "mmaFrag"; readonly acc: string;
      readonly a: FragExpr; readonly b: Expr };

/** A fragment-valued expression: an accumulator, or an elementwise map of one. */
export type FragExpr =
  | { readonly k: "acc"; readonly name: string }
  | { readonly k: "map"; readonly op: "relu"; readonly a: FragExpr }
  /**
   * The same elementwise and row-scaling vocabulary `Expr` has for staged tiles,
   * over a computed fragment instead. Fusing a row-wise normalisation onto a
   * contraction needs it — `expTile(subRow(s, mx))` where `s` came out of `mma`
   * rather than out of memory — and until now a fragment could only be stored or
   * relu'd, because nothing else was ever done to one.
   */
  | { readonly k: "unary"; readonly op: "sq" | "exp"; readonly a: FragExpr }
  | { readonly k: "rowOp"; readonly op: "sub" | "mul" | "div";
      readonly a: FragExpr; readonly row: RowExpr };

export type Pass =
  /**
   * `locals` and `inner` are how one contraction sits inside another's loop.
   *
   * `for n { let s = zeros(...); for k { s = mma(...) } ; mx = rowMax(s, mx) }` —
   * the scores for one block of `n` are computed by an inner contraction over
   * `k`, folded into the outer accumulator, and thrown away. The fragment is a
   * temporary within one outer iteration, which is the easier half of composing
   * contractions: nothing is carried across, so nothing has to survive a change
   * of register layout.
   *
   * Both are empty for a flat pass, which is every kernel written before this.
   */
  | { readonly k: "reduce"; readonly axis: string;
      readonly locals: readonly Acc[]; readonly inner: readonly Pass[];
      readonly updates: readonly Update[] }
  /** A store inside a pass, one operand cell at a time. */
  | { readonly k: "store"; readonly axis: string; readonly binding: string;
      readonly locals: readonly Acc[]; readonly inner: readonly Pass[];
      readonly coords: readonly Coord[];
      /** Which of the two `value` is. A store inside a pass that ran an inner
       *  contraction writes the fragment that contraction built. */
      readonly fromFrag: boolean; readonly value: Expr | FragExpr };

/**
 * A step in the body, in SOURCE ORDER.
 *
 * Passes and derived row values interleave — layernorm reduces, then computes a
 * mean and a reciprocal standard deviation from the accumulators, then stores.
 * An earlier version kept them in two lists and lost that ordering, and the
 * emitter then referenced `mu` before anything declared it. WGSL caught it;
 * keeping the order is the fix.
 */
/**
 * One argument of `.tile(...)`.
 *
 * These used to be discarded. With `spec.grid` and `spec.reduce` declared, the
 * coordinate system was implied by the schedule and the IR never had to know
 * which axis went where — transposition was caught by the type checker alone,
 * and nothing downstream could see it. Deriving the axis roles from the body
 * means reading them: `at.m` is a free coordinate, a loop variable is a
 * contracted one, and the difference is the whole question.
 */
export type Coord =
  | { readonly kind: "at"; readonly axis: string }
  | { readonly kind: "loop"; readonly name: string };

export type Step =
  | { readonly k: "derived"; readonly name: string; readonly expr: RowExpr }
  /** A fragment-valued intermediate, e.g. `const e = expTile(subRow(s, mx));` */
  | { readonly k: "derivedFrag"; readonly name: string; readonly expr: FragExpr }
  /** A store OUTSIDE any pass, of a whole fragment. matmul ends this way. */
  | { readonly k: "storeFrag"; readonly binding: string;
      readonly coords: readonly Coord[]; readonly value: FragExpr }
  | Pass;

export interface Acc {
  readonly name: string;
  readonly kind: AccKind;
  readonly init: PadName;
}

/**
 * A kernel body, parsed.
 *
 * One representation for both schedules. What differs is the accumulator kind and
 * therefore the update form — a row accumulator folds one cell at a time, a
 * fragment accumulates an outer product — and where the store sits. Both are read
 * from the source rather than matched against a shape.
 */
export interface Body {
  readonly accKind: AccKind;
  readonly accs: readonly Acc[];
  readonly steps: readonly Step[];
}

/** Retained for the row-wise emitter's narrower expectations. */
export type RowBody = Body;

const BLOCK_UNARY: Record<string, "sq" | "exp"> = { sqTile: "sq", expTile: "exp" };
const ROW_COMBINE: Record<string, "sub" | "mul" | "div"> = {
  subRow: "sub", mulRow: "mul", divRow: "div",
};
const REDUCERS: Record<string, "max" | "sum"> = { rowMax: "max", rowSum: "sum" };
const FRAG_MAP: Record<string, "relu"> = { relu: "relu" };

/**
 * Does this expression ultimately read a fragment?
 *
 * `expTile`, `subRow` and friends apply to a staged tile and to a computed
 * fragment alike — the spelling is the same and the operand decides. So the
 * parser follows the chain down to a leaf and asks what the leaf is.
 */
function bottomsOutAtFrag(n: ts.Expression, fragNames: ReadonlySet<string>): boolean {
  if (ts.isIdentifier(n)) return fragNames.has(n.text);
  if (!ts.isCallExpression(n) || !n.arguments.length) return false;
  const fn = callee(n);
  // A REDUCER consumes a fragment and produces a row, so the leaf does not decide
  // for it: `rowMax(s, …)` bottoms out at a fragment and is a row value anyway.
  if (fn && fn in REDUCERS) return false;
  return bottomsOutAtFrag(n.arguments[0], fragNames);
}

/** `at.<axis>` or a loop variable. Anything else is not a coordinate. */
export function readCoord(n: ts.Expression): Coord {
  if (ts.isPropertyAccessExpression(n) && ts.isIdentifier(n.expression)
      && n.expression.text === "at") {
    return { kind: "at", axis: n.name.text };
  }
  if (ts.isIdentifier(n)) return { kind: "loop", name: n.text };
  throw new BodyError(
    `a tile coordinate is \`at.<axis>\` or a loop variable, got \`${n.getText()}\``, n);
}

export class BodyError extends Error {
  constructor(message: string, readonly node: ts.Node) { super(message); }
}

const bad: (n: ts.Node, m: string) => never = (n, m) => { throw new BodyError(m, n); };

const callee = (n: ts.CallExpression): string | undefined =>
  ts.isIdentifier(n.expression) ? n.expression.text
    : ts.isPropertyAccessExpression(n.expression) ? n.expression.name.text
    : undefined;

/**
 * Parse a block-valued expression. Every accepted form is listed here; anything
 * else is an error naming what was seen, never a silent approximation.
 */
export function parseExpr(
  n: ts.Expression,
  rowNames: ReadonlySet<string>,
  bindings: ReadonlySet<string>,
  padNames: readonly string[],
): Expr {
  if (!ts.isCallExpression(n)) {
    bad(n, `expected a block expression, got ${ts.SyntaxKind[n.kind]}`);
  }
  const fn = callee(n);
  if (!fn) bad(n, `cannot tell what is being called here`);

  if (fn === "pad") {
    const recv = (n.expression as ts.PropertyAccessExpression).expression;
    const inner = parseExpr(recv, rowNames, bindings, padNames);
    if (inner.k !== "tile") bad(n, `.pad() applies to a tile, not to ${inner.k}`);
    const arg = n.arguments[0];
    const id = arg && ts.isIdentifier(arg) ? arg.text : undefined;
    if (!id || !padNames.includes(id)) {
      bad(arg ?? n,
        `.pad() takes a named identity — one of ${padNames.join(", ")} — not a number. ` +
        `The identity for a masked max is negative infinity, which TypeScript cannot ` +
        `express as a literal type.`);
    }
    return {
      k: "tile", binding: inner.binding, coords: inner.coords,
      transposed: inner.transposed, pad: id as PadName,
    };
  }

  if (fn === "tile" || fn === "tileT") {
    const recv = (n.expression as ts.PropertyAccessExpression).expression;
    if (!ts.isIdentifier(recv) || !bindings.has(recv.text)) {
      bad(n, `.${fn}() must be called on a binding declared in spec.bindings`);
    }
    return {
      k: "tile", binding: recv.text, transposed: fn === "tileT",
      coords: n.arguments.map((a) => readCoord(a)),
    };
  }

  if (fn in BLOCK_UNARY) {
    return { k: "unary", op: BLOCK_UNARY[fn], a: parseExpr(n.arguments[0], rowNames, bindings, padNames) };
  }

  if (fn in ROW_COMBINE) {
    return {
      k: "rowOp", op: ROW_COMBINE[fn],
      a: parseExpr(n.arguments[0], rowNames, bindings, padNames),
      row: parseRow(n.arguments[1], rowNames),
    };
  }

  return bad(n, `'${fn}' is not a block operation this build knows. Known: ` +
    `tile, pad, ${Object.keys(BLOCK_UNARY).join(", ")}, ${Object.keys(ROW_COMBINE).join(", ")}.`);
}

/** Parse a row-valued expression: an accumulator, or something derived from one. */
export function parseRow(n: ts.Expression, rowNames: ReadonlySet<string>): RowExpr {
  if (ts.isIdentifier(n)) {
    if (!rowNames.has(n.text)) bad(n, `'${n.text}' is not a row value in scope`);
    return { k: "acc", name: n.text };
  }
  if (!ts.isCallExpression(n)) bad(n, `expected a row value, got ${ts.SyntaxKind[n.kind]}`);
  const fn = callee(n);
  if (fn === "rowMax" || fn === "rowSum") {
    // `rowMax(frag, rowFill(bm, f32, negInf))` — the accumulator is written
    // inline because there is nothing to accumulate ACROSS: one fragment, one
    // reduction. Inside a loop the same call is an update instead, read by
    // readUpdatesAndStore.
    const acc = n.arguments[1];
    if (!acc || !ts.isCallExpression(acc) || callee(acc) !== "rowFill") {
      bad(acc ?? n,
        `outside a reduce loop, ${fn}'s accumulator is written inline as ` +
        `rowFill(rows, dtype, identity) — there is no earlier pass to carry one`);
    }
    const id = acc.arguments[2];
    const nm = id && ts.isIdentifier(id) ? id.text : undefined;
    if (!nm) bad(id ?? acc, `rowFill's identity must be a named identity`);
    return {
      k: "reduceFrag", op: fn === "rowMax" ? "max" : "sum",
      a: parseFrag(n.arguments[0], rowNames), init: nm as PadName,
    };
  }
  if (fn === "meanRow") return { k: "mean", a: parseRow(n.arguments[0], rowNames) };
  if (fn === "rstdRow") {
    const eps = n.arguments[2];
    if (!eps || !ts.isNumericLiteral(eps)) {
      bad(eps ?? n, `rstdRow's epsilon must be a numeric literal — it is folded into the shader`);
    }
    return {
      k: "rstd",
      sumSq: parseRow(n.arguments[0], rowNames),
      mean: parseRow(n.arguments[1], rowNames),
      eps: Number(eps.text),
    };
  }
  return bad(n, `'${fn}' is not a row operation this build knows. ` +
                `Known: rowMax, rowSum, meanRow, rstdRow.`);
}

/**
 * Read a row-wise kernel body into a RowBody.
 *
 * The statement grammar, exhaustively:
 *
 *   let <acc> = rowFill(<rows>, <dtype>, <identity>);   an accumulator
 *   const <name> = <row expression>;                    a derived row value
 *   for (const <i> of reduce.<axis>) { ... }            a pass over the reduce axis
 *       <acc> = rowMax|rowSum(<block expr>, <acc>);       an update, inside a pass
 *       <binding>.tile(...).store(<block expr>);          the store, inside a pass
 *
 * Nothing here counts operators or matches a per-kernel template. How many
 * accumulators exist, how many passes there are and what each one does all come
 * out of the statements. softmax is 2 accumulators over 3 passes; layernorm is 2
 * over 2, updating both in the first; neither shape is written down anywhere.
 */
/** A fragment-valued expression: an accumulator, or an elementwise map of one. */
export function parseFrag(n: ts.Expression, rowNames: ReadonlySet<string>): FragExpr {
  if (ts.isIdentifier(n)) {
    if (!rowNames.has(n.text)) bad(n, `'${n.text}' is not an accumulator in scope`);
    return { k: "acc", name: n.text };
  }
  if (!ts.isCallExpression(n)) bad(n, `expected a fragment, got ${ts.SyntaxKind[n.kind]}`);
  const fn = callee(n);
  if (fn && fn in FRAG_MAP) return { k: "map", op: FRAG_MAP[fn], a: parseFrag(n.arguments[0], rowNames) };
  if (fn && fn in BLOCK_UNARY) {
    return { k: "unary", op: BLOCK_UNARY[fn], a: parseFrag(n.arguments[0], rowNames) };
  }
  if (fn && fn in ROW_COMBINE) {
    return {
      k: "rowOp", op: ROW_COMBINE[fn],
      a: parseFrag(n.arguments[0], rowNames),
      row: parseRow(n.arguments[1], rowNames),
    };
  }
  return bad(n, `'${fn}' is not a fragment operation. Known: ` +
                `${[...Object.keys(FRAG_MAP), ...Object.keys(BLOCK_UNARY), ...Object.keys(ROW_COMBINE)].join(", ")}.`);
}

export function parseBody(
  body: ts.ConciseBody,
  bindings: ReadonlySet<string>,
  padNames: readonly string[],
): Body {
  if (!ts.isBlock(body)) bad(body, `a kernel body must be a block`);

  const accs: Acc[] = [];
  const fragNames = new Set<string>();
  const steps: Step[] = [];
  const rowNames = new Set<string>();

  const readUpdatesAndStore = (block: ts.Statement, axis: string): Pass => {
    if (!ts.isBlock(block)) bad(block, `a pass body must be a block`);
    const updates: Update[] = [];
    const locals: Acc[] = [];
    const inner: Pass[] = [];
    let store: { binding: string; coords: readonly Coord[];
                 fromFrag: boolean; value: Expr | FragExpr } | undefined;

    for (const st of block.statements) {
      // A LOCAL accumulator: the scores for this block of the outer axis, which
      // the inner contraction fills and the outer fold consumes. Scoped to the
      // pass, unlike the accumulators declared at the top of the body, because it
      // does not survive the iteration.
      if (ts.isVariableStatement(st)) {
        for (const d of st.declarationList.declarations) {
          if (!ts.isIdentifier(d.name)) bad(d, `destructuring is not admitted here`);
          const init = d.initializer;
          if (!init || !ts.isCallExpression(init) || callee(init) !== "zeros") {
            bad(d, `inside a pass, a declaration is a fragment accumulator: ` +
                   `\`let s = zeros(bm, bn, dtype)\``);
          }
          locals.push({ name: d.name.text, kind: "frag", init: "zero" });
          rowNames.add(d.name.text);
          fragNames.add(d.name.text);
        }
        continue;
      }

      // A NESTED contraction. `for n { … for k { … } … }`: the inner pass runs to
      // completion for each step of the outer one, which is what makes composing
      // contractions possible at all — the inner fragment is whole before the
      // outer fold reads it.
      if (ts.isForOfStatement(st)) {
        const it = st.expression;
        if (!ts.isPropertyAccessExpression(it) || !ts.isIdentifier(it.expression)
            || it.expression.text !== "reduce") {
          bad(st, `a nested pass iterates \`reduce.<axis>\`, got \`${it.getText()}\``);
        }
        if (it.name.text === axis) {
          bad(st, `this pass already reduces over "${axis}"; a nested pass must ` +
                  `contract a different axis`);
        }
        inner.push(readUpdatesAndStore(st.statement, it.name.text));
        continue;
      }

      if (ts.isExpressionStatement(st) && ts.isBinaryExpression(st.expression)
          && st.expression.operatorToken.kind === ts.SyntaxKind.EqualsToken) {
        const lhs = st.expression.left, rhs = st.expression.right;
        if (!ts.isIdentifier(lhs) || !rowNames.has(lhs.text)) {
          bad(lhs, `only a declared accumulator may be assigned inside a pass`);
        }
        if (!ts.isCallExpression(rhs)) bad(rhs, `expected a reduction`);
        const fn = callee(rhs);
        if (!fn || !(fn in REDUCERS || fn === "mma")) {
          bad(rhs, `'${fn}' is not a reduction. Known: ${Object.keys(REDUCERS).join(", ")}, mma.`);
        }
        // The accumulator is the last argument in both forms.
        const carried = rhs.arguments[fn === "mma" ? 2 : 1];
        if (!carried || !ts.isIdentifier(carried) || carried.text !== lhs.text) {
          bad(carried ?? rhs,
            `a reduction must carry the accumulator it assigns — write ` +
            `\`${lhs.text} = ${fn}(..., ${lhs.text})\`. Carrying a different one silently ` +
            `computes something else.`);
        }
        if (fn === "mma") {
          updates.push({
            ...(bottomsOutAtFrag(rhs.arguments[0], fragNames)
              ? { k: "mmaFrag" as const, acc: lhs.text,
                  a: parseFrag(rhs.arguments[0], rowNames),
                  b: parseExpr(rhs.arguments[1], rowNames, bindings, padNames) }
              : { k: "mma" as const, acc: lhs.text,
                  a: parseExpr(rhs.arguments[0], rowNames, bindings, padNames),
                  b: parseExpr(rhs.arguments[1], rowNames, bindings, padNames) }),
          });
        } else {
          updates.push(bottomsOutAtFrag(rhs.arguments[0], fragNames)
            ? { k: "foldFrag", acc: lhs.text, op: REDUCERS[fn],
                value: parseFrag(rhs.arguments[0], rowNames) }
            : { k: "fold", acc: lhs.text, op: REDUCERS[fn],
                value: parseExpr(rhs.arguments[0], rowNames, bindings, padNames) });
        }
        continue;
      }

      if (ts.isExpressionStatement(st) && ts.isCallExpression(st.expression)
          && callee(st.expression) === "store") {
        const slot = (st.expression.expression as ts.PropertyAccessExpression).expression;
        if (!ts.isCallExpression(slot) || callee(slot) !== "tile") {
          bad(st, `a store goes through <binding>.tile(...).store(...)`);
        }
        const recv = (slot.expression as ts.PropertyAccessExpression).expression;
        if (!ts.isIdentifier(recv) || !bindings.has(recv.text)) {
          bad(st, `.store() must go to a binding declared in spec.bindings`);
        }
        if (store) bad(st, `more than one store in a pass`);
        const sarg = st.expression.arguments[0];
        const fromFrag = bottomsOutAtFrag(sarg, fragNames);
        store = {
          binding: recv.text,
          coords: slot.arguments.map((a) => readCoord(a)),
          fromFrag,
          value: fromFrag ? parseFrag(sarg, rowNames)
                          : parseExpr(sarg, rowNames, bindings, padNames),
        };
        continue;
      }

      bad(st, `TSA0104: ${ts.SyntaxKind[st.kind]} is not admitted inside a pass`);
    }

    if (store && updates.length) bad(block, `a pass either reduces or stores, not both`);
    if (store) return { k: "store", axis, locals, inner, ...store };
    if (!updates.length) bad(block, `this pass does nothing`);
    return { k: "reduce", axis, locals, inner, updates };
  };

  for (const st of body.statements) {
    if (ts.isVariableStatement(st)) {
      for (const d of st.declarationList.declarations) {
        if (!ts.isIdentifier(d.name)) bad(d, `destructuring is not admitted here`);
        const init = d.initializer;
        if (!init || !ts.isCallExpression(init)) bad(d, `a declaration needs a call initialiser`);

        if (callee(init) === "rowFill") {
          const id = init.arguments[2];
          const nm = id && ts.isIdentifier(id) ? id.text : undefined;
          if (!nm || !padNames.includes(nm)) {
            bad(id ?? init, `rowFill's identity must be one of ${padNames.join(", ")}`);
          }
          accs.push({ name: d.name.text, kind: "row", init: nm as PadName });
        } else if (callee(init) === "zeros") {
          // A 2-D register fragment. `zeros` names its identity by being zeros.
          accs.push({ name: d.name.text, kind: "frag", init: "zero" });
          fragNames.add(d.name.text);
        } else if (bottomsOutAtFrag(init, fragNames)) {
          // `const e = expTile(subRow(s, mx))` where `s` is a fragment. The same
          // call spelling applies to tiles, so the operand decides, not the name.
          steps.push({ k: "derivedFrag", name: d.name.text, expr: parseFrag(init, rowNames) });
          fragNames.add(d.name.text);
        } else {
          steps.push({ k: "derived", name: d.name.text, expr: parseRow(init, rowNames) });
        }
        rowNames.add(d.name.text);
      }
      continue;
    }

    if (ts.isForOfStatement(st)) {
      // Which axis this pass reduces over used to be discarded: there was exactly
      // one reduce axis for the whole kernel, so `for (const n of reduce.n)` had
      // nothing to distinguish. It is read now because the axis roles come from
      // the body rather than from a declaration — an axis may be contracted in
      // one pass and free in another, which is what attention needs.
      const it = st.expression;
      if (!ts.isPropertyAccessExpression(it) || !ts.isIdentifier(it.expression)
          || it.expression.text !== "reduce") {
        bad(st, `a pass iterates \`reduce.<axis>\`, got \`${it.getText()}\``);
      }
      steps.push(readUpdatesAndStore(st.statement, it.name.text));
      continue;
    }

    // `c.tile(...).store(relu(acc))` outside any loop: a whole-fragment store.
    if (ts.isExpressionStatement(st) && ts.isCallExpression(st.expression)
        && callee(st.expression) === "store") {
      const slot = (st.expression.expression as ts.PropertyAccessExpression).expression;
      if (!ts.isCallExpression(slot) || callee(slot) !== "tile") {
        bad(st, `a store goes through <binding>.tile(...).store(...)`);
      }
      const recv = (slot.expression as ts.PropertyAccessExpression).expression;
      if (!ts.isIdentifier(recv) || !bindings.has(recv.text)) {
        bad(st, `.store() must go to a binding declared in spec.bindings`);
      }
      steps.push({
        k: "storeFrag", binding: recv.text,
        coords: slot.arguments.map((a) => readCoord(a)),
        value: parseFrag(st.expression.arguments[0], rowNames),
      });
      continue;
    }

    bad(st, `TSA0104: ${ts.SyntaxKind[st.kind]} is not admitted at the top level of a body`);
  }

  if (!accs.length) bad(body, `no accumulator: a body starts with zeros(...) or rowFill(...)`);
  if (!steps.some((p) => p.k === "store" || p.k === "storeFrag")) {
    bad(body, `no store: nothing is written`);
  }
  const kinds = new Set(accs.map((a) => a.kind));
  if (kinds.size > 1) bad(body, `a body mixes register fragments with row accumulators`);
  return { accKind: accs[0].kind, accs, steps };
}
