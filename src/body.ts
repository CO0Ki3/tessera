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
  | { readonly k: "tile"; readonly binding: string; readonly pad?: PadName }
  | { readonly k: "unary"; readonly op: "sq" | "exp"; readonly a: Expr }
  /** A block combined with a row value, broadcast along the block's columns. */
  | { readonly k: "rowOp"; readonly op: "sub" | "mul" | "div"; readonly a: Expr; readonly row: RowExpr };

/** A row-valued expression: one value per row the workgroup owns. */
export type RowExpr =
  | { readonly k: "acc"; readonly name: string }
  | { readonly k: "mean"; readonly a: RowExpr }
  | { readonly k: "rstd"; readonly sumSq: RowExpr; readonly mean: RowExpr; readonly eps: number };

/** How an accumulator is laid out per invocation. */
export type AccKind = "row" | "frag";

export type Update =
  /** Fold one operand cell into a row accumulator. */
  | { readonly k: "fold"; readonly acc: string; readonly op: "max" | "sum"; readonly value: Expr }
  /** Outer-product two operand slices into a 2-D fragment. */
  | { readonly k: "mma"; readonly acc: string; readonly a: Expr; readonly b: Expr };

/** A fragment-valued expression: an accumulator, or an elementwise map of one. */
export type FragExpr =
  | { readonly k: "acc"; readonly name: string }
  | { readonly k: "map"; readonly op: "relu"; readonly a: FragExpr };

export type Pass =
  | { readonly k: "reduce"; readonly updates: readonly Update[] }
  /** A store inside a pass, one operand cell at a time. */
  | { readonly k: "store"; readonly binding: string; readonly value: Expr };

/**
 * A step in the body, in SOURCE ORDER.
 *
 * Passes and derived row values interleave — layernorm reduces, then computes a
 * mean and a reciprocal standard deviation from the accumulators, then stores.
 * An earlier version kept them in two lists and lost that ordering, and the
 * emitter then referenced `mu` before anything declared it. WGSL caught it;
 * keeping the order is the fix.
 */
export type Step =
  | { readonly k: "derived"; readonly name: string; readonly expr: RowExpr }
  /** A store OUTSIDE any pass, of a whole fragment. matmul ends this way. */
  | { readonly k: "storeFrag"; readonly binding: string; readonly value: FragExpr }
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
    return { k: "tile", binding: inner.binding, pad: id as PadName };
  }

  if (fn === "tile") {
    const recv = (n.expression as ts.PropertyAccessExpression).expression;
    if (!ts.isIdentifier(recv) || !bindings.has(recv.text)) {
      bad(n, `.tile() must be called on a binding declared in spec.bindings`);
    }
    return { k: "tile", binding: recv.text };
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
  return bad(n, `'${fn}' is not a row operation this build knows. Known: meanRow, rstdRow.`);
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
  return bad(n, `'${fn}' is not a fragment operation. Known: ${Object.keys(FRAG_MAP).join(", ")}.`);
}

export function parseBody(
  body: ts.ConciseBody,
  bindings: ReadonlySet<string>,
  padNames: readonly string[],
): Body {
  if (!ts.isBlock(body)) bad(body, `a kernel body must be a block`);

  const accs: Acc[] = [];
  const steps: Step[] = [];
  const rowNames = new Set<string>();

  const readUpdatesAndStore = (block: ts.Statement): Pass => {
    if (!ts.isBlock(block)) bad(block, `a pass body must be a block`);
    const updates: Update[] = [];
    let store: { binding: string; value: Expr } | undefined;

    for (const st of block.statements) {
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
            k: "mma", acc: lhs.text,
            a: parseExpr(rhs.arguments[0], rowNames, bindings, padNames),
            b: parseExpr(rhs.arguments[1], rowNames, bindings, padNames),
          });
        } else {
          updates.push({
            k: "fold", acc: lhs.text, op: REDUCERS[fn],
            value: parseExpr(rhs.arguments[0], rowNames, bindings, padNames),
          });
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
        store = {
          binding: recv.text,
          value: parseExpr(st.expression.arguments[0], rowNames, bindings, padNames),
        };
        continue;
      }

      bad(st, `TSA0104: ${ts.SyntaxKind[st.kind]} is not admitted inside a pass`);
    }

    if (store && updates.length) bad(block, `a pass either reduces or stores, not both`);
    if (store) return { k: "store", ...store };
    if (!updates.length) bad(block, `this pass does nothing`);
    return { k: "reduce", updates };
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
        } else {
          steps.push({ k: "derived", name: d.name.text, expr: parseRow(init, rowNames) });
        }
        rowNames.add(d.name.text);
      }
      continue;
    }

    if (ts.isForOfStatement(st)) { steps.push(readUpdatesAndStore(st.statement)); continue; }

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
