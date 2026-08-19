/**
 * frontend.ts — TypeScript source in, tessera IR out.
 *
 * The load-bearing idea: tessera reads the kernel spec from TYPES, not from call
 * arguments. In the canonical kernel the block size is written `T.bm`, not `64`:
 *
 *     const T = tiling(f32, 64, 64, 16);
 *     const M = axis("m", 1024, T.bm);
 *
 * An argument-parsing front end would need its own constant propagation to learn
 * that the block is 64. The checker already did that work, and says so:
 * `Axis<"m", 1024, 64, "exact">`. So we ask the checker.
 *
 * That is not a convenience. It is the project's thesis executed once: the type
 * system is where the shape information lives, and everything downstream — the
 * masks, the fragment size, the dispatch extents — is folded from it.
 */

import ts from "typescript";
import {
  derive, type AxisIR, type BindingIR, type DTypeName, type KernelIR, type PadName, type Schedule,
} from "./ir.ts";

import {
  parseBody, BodyError, type Body, type RowBody, type Expr,
} from "./body.ts";

const PAD_NAMES = ["zero", "one", "negInf", "posInf"] as const;

export class FrontendError extends Error {
  constructor(message: string, readonly node?: ts.Node) {
    super(message);
  }
}

function fail(node: ts.Node | undefined, message: string): never {
  if (node) {
    const sf = node.getSourceFile();
    const { line, character } = sf.getLineAndCharacterOfPosition(node.getStart());
    throw new FrontendError(`${sf.fileName}:${line + 1}:${character + 1}  ${message}`, node);
  }
  throw new FrontendError(message);
}

// ---------------------------------------------------------------------------
// Reading literals back out of types
// ---------------------------------------------------------------------------

function propType(checker: ts.TypeChecker, t: ts.Type, name: string): ts.Type | undefined {
  const sym = t.getProperty(name);
  return sym ? checker.getTypeOfSymbol(sym) : undefined;
}

function numberLiteral(checker: ts.TypeChecker, t: ts.Type, prop: string, where: string): number {
  const pt = propType(checker, t, prop);
  if (!pt || !pt.isNumberLiteral()) {
    fail(undefined,
      `${where}: '${prop}' is ${pt ? checker.typeToString(pt) : "absent"}, not a numeric literal. ` +
      `Every extent, block and tile dimension must be a compile-time integer literal.`);
  }
  return pt.value;
}

function stringLiteral(checker: ts.TypeChecker, t: ts.Type, prop: string, where: string): string {
  const pt = propType(checker, t, prop);
  if (!pt || !pt.isStringLiteral()) {
    fail(undefined, `${where}: '${prop}' is ${pt ? checker.typeToString(pt) : "absent"}, not a string literal.`);
  }
  return pt.value;
}

function tupleElements(checker: ts.TypeChecker, t: ts.Type): ts.Type[] {
  if (checker.isTupleType(t)) return [...checker.getTypeArguments(t as ts.TypeReference)];
  if (checker.isArrayType(t)) {
    // A readonly array that did not stay a tuple means a `const` assertion was
    // lost somewhere; the element count is then unknown at compile time.
    fail(undefined, `expected a fixed-length tuple, got ${checker.typeToString(t)}`);
  }
  fail(undefined, `expected a tuple, got ${checker.typeToString(t)}`);
}

function readAxis(checker: ts.TypeChecker, t: ts.Type, where: string): AxisIR {
  const name = stringLiteral(checker, t, "name", where);
  const extent = numberLiteral(checker, t, "extent", `${where} axis "${name}"`);
  const block = numberLiteral(checker, t, "block", `${where} axis "${name}"`);
  const fitT = propType(checker, t, "fit");
  if (!fitT || !fitT.isStringLiteral() || (fitT.value !== "exact" && fitT.value !== "ragged")) {
    fail(undefined, `${where} axis "${name}": 'fit' is not "exact" | "ragged"`);
  }
  const fit = fitT.value as "exact" | "ragged";

  if (fit === "exact" && extent % block !== 0) {
    const legal = [8, 16, 32, 64, 128].filter((b) => extent % b === 0);
    fail(undefined,
      `TSA0301: axis "${name}" has extent ${extent}, which ${block} does not divide. ` +
      (legal.length
        ? `Blocks that do divide ${extent}: ${legal.join(", ")}. `
        : `No standard block divides ${extent}. `) +
      `Use raggedAxis("${name}", ${extent}, ${block}) to opt into masking.`);
  }

  return { name, extent, block, fit, tiles: Math.ceil(extent / block) };
}

function readDType(checker: ts.TypeChecker, t: ts.Type, where: string): DTypeName {
  const d = stringLiteral(checker, t, "dtype", where);
  if (d !== "f32" && d !== "f16" && d !== "i32") fail(undefined, `${where}: unknown dtype '${d}'`);
  return d;
}

// ---------------------------------------------------------------------------
// Finding the kernel() call
// ---------------------------------------------------------------------------

function findKernelCall(sf: ts.SourceFile, checker: ts.TypeChecker): ts.CallExpression {
  const found: ts.CallExpression[] = [];
  const visit = (n: ts.Node): void => {
    if (ts.isCallExpression(n) && ts.isIdentifier(n.expression) && n.expression.text === "kernel") {
      found.push(n);
    }
    ts.forEachChild(n, visit);
  };
  visit(sf);

  if (found.length === 0) fail(undefined, `${sf.fileName}: no kernel(...) call found`);
  if (found.length > 1) {
    fail(found[1], `more than one kernel(...) per file is not supported yet (found ${found.length})`);
  }
  return found[0];
}

function readSpecName(specNode: ts.Expression): string {
  if (!ts.isObjectLiteralExpression(specNode)) {
    fail(specNode, "kernel()'s spec must be written as an object literal");
  }
  for (const p of specNode.properties) {
    if (ts.isPropertyAssignment(p) && ts.isIdentifier(p.name) && p.name.text === "name") {
      if (!ts.isStringLiteral(p.initializer)) {
        fail(p.initializer, "spec.name must be a string literal — it becomes the entry point name");
      }
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(p.initializer.text)) {
        fail(p.initializer,
          `spec.name "${p.initializer.text}" is not a valid identifier; it is emitted as a ` +
          `WGSL entry point name`);
      }
      return p.initializer.text;
    }
  }
  fail(specNode, "spec.name is missing");
}

/**
 * Confirm the body is the canonical shape this walking skeleton knows how to
 * emit, and say plainly when it is not.
 *
 * Compiling an arbitrary admitted body is the next milestone. Silently accepting
 * a body we do not actually read would be exactly the "fell off the fast path"
 * failure the project's admission rule forbids, so this is a hard error.
 */
function checkCanonicalBody(
  call: ts.CallExpression,
  bindings: readonly BindingIR[],
  raggedNames: ReadonlySet<string>,
): { padded: Map<string, PadName>; schedule: Schedule; rowBody: RowBody } {
  const body = call.arguments[1];
  if (!body || (!ts.isArrowFunction(body) && !ts.isFunctionExpression(body))) {
    fail(body ?? call, "kernel()'s second argument must be a function");
  }

  // A DEFAULT-REJECT whitelist, not a count of the things we expect to find.
  // Counting is default-accept: an earlier version of this function counted
  // zeros/loops/mma/store and happily compiled a body containing `while (false)
  // {}`, emitting code that was not what the user wrote. That is precisely the
  // silent fall-off the project's admission rule forbids, so the arm below
  // rejects every kind not named here.
  const ADMITTED = new Set<ts.SyntaxKind>([
    ts.SyntaxKind.Block,
    ts.SyntaxKind.VariableStatement,
    ts.SyntaxKind.VariableDeclarationList,
    ts.SyntaxKind.VariableDeclaration,
    ts.SyntaxKind.ExpressionStatement,
    ts.SyntaxKind.ForOfStatement,
    ts.SyntaxKind.CallExpression,
    ts.SyntaxKind.PropertyAccessExpression,
    ts.SyntaxKind.BinaryExpression,   // narrowed to `=` below
    ts.SyntaxKind.Identifier,
  ]);

  // Tokens carry no admission decision of their own — the node that owns them is
  // already constrained, and a token has no children. Two are hazards in themselves.
  const DENIED_TOKENS = new Set<ts.SyntaxKind>([
    ts.SyntaxKind.ThisKeyword,
    ts.SyntaxKind.SuperKeyword,
    ts.SyntaxKind.ImportKeyword,
  ]);

  const padded = new Map<string, PadName>();

  const visit = (n: ts.Node): void => {
    if (ts.isToken(n)) {
      if (DENIED_TOKENS.has(n.kind)) {
        fail(n, `TSA0015: '${ts.SyntaxKind[n.kind].replace("Keyword", "").toLowerCase()}' ` +
                `is not admitted inside a kernel body.`);
      }
      return;
    }
    if (!ADMITTED.has(n.kind)) {
      fail(n,
        `TSA0104: ${ts.SyntaxKind[n.kind]} is not admitted inside a kernel body. ` +
        `This build emits a fixed set of schedules; rather than silently compiling ` +
        `something that is not what you wrote, it refuses.`);
    }
    if (ts.isBinaryExpression(n) && n.operatorToken.kind !== ts.SyntaxKind.EqualsToken) {
      fail(n,
        `TSA0016: only whole-expression assignment is admitted in a kernel body, ` +
        `not '${ts.tokenToString(n.operatorToken.kind)}'.`);
    }

    if (ts.isCallExpression(n) && ts.isPropertyAccessExpression(n.expression)) {
      const method = n.expression.name.text;
      if (method === "pad") {
        const inner = n.expression.expression;
        const owner = ts.isCallExpression(inner) && ts.isPropertyAccessExpression(inner.expression)
          && ts.isIdentifier(inner.expression.expression)
          ? inner.expression.expression.text : undefined;
        if (!owner || !bindings.some((b) => b.name === owner)) {
          fail(n, `.pad() must be called on a binding's tile, as <binding>.tile(...).pad(zero)`);
        }
        // Which identity an operator ACCEPTS is enforced by the surface types;
        // this only records which one was named.
        const arg = n.arguments[0];
        const named = arg && ts.isIdentifier(arg) ? arg.text : undefined;
        if (!named || !(PAD_NAMES as readonly string[]).includes(named)) {
          fail(arg ?? n,
            `.pad() takes a named identity element — one of ${PAD_NAMES.join(", ")} — ` +
            `not a number. The identity for a masked max is negative infinity, which ` +
            `TypeScript cannot express as a literal type.`);
        }
        padded.set(owner, named as PadName);
      }
    }
    ts.forEachChild(n, visit);
  };
  visit(body.body);

  // ---- read the body ------------------------------------------------------
  // Not "which template does this match". The body is parsed into accumulators,
  // steps and a small expression IR; the schedule then falls out of what kind of
  // accumulator it declares. Counting operator names -- which is what this did
  // for two kernel families and would have done for a third -- is gone.
  let parsed: Body;
  try {
    parsed = parseBody(body.body, new Set(bindings.map((b) => b.name)), PAD_NAMES);
  } catch (e) {
    if (e instanceof BodyError) fail(e.node, e.message);
    throw e;
  }

  const schedule: Schedule = parsed.accKind === "frag" ? "matmul" : "rowwise";

  // The two emitters still differ, and honestly so: a fragment accumulates an
  // outer product from two staged operands, a row accumulator folds one cell at a
  // time from an operand read straight from global memory. Unifying those means a
  // contraction abstraction, which is a design step rather than a refactor.
  if (schedule === "matmul") {
    if (!parsed.steps.some((x) => x.k === "storeFrag")) {
      fail(body, `a fragment schedule ends by storing the fragment`);
    }
  }

  return { padded, schedule, rowBody: parsed };
}

// ---------------------------------------------------------------------------

/**
 * Which bindings the body reads through `.tileT()`.
 *
 * The bindings are built from the spec before the body is parsed, so this runs
 * afterwards and fills the flag in. A binding read BOTH ways in one body would
 * need two staging layouts for one buffer, which the emitter does not do, so it
 * is refused rather than silently given one of them.
 */
function collectTransposed(parsed: RowBody, node: ts.Node): Set<string> {
  const t = new Set<string>(), plain = new Set<string>();
  const walk = (e: Expr): void => {
    if (e.k === "tile") { (e.transposed ? t : plain).add(e.binding); return; }
    if (e.k === "unary") return walk(e.a);
    if (e.k === "rowOp") return walk(e.a);
  };
  for (const st of parsed.steps) {
    if (st.k === "store") walk(st.value);
    if (st.k === "reduce") {
      for (const u of st.updates) {
        if (u.k === "fold") walk(u.value);
        if (u.k === "mma") { walk(u.a); walk(u.b); }
      }
    }
  }
  for (const n of t) {
    if (plain.has(n)) {
      fail(node, `binding "${n}" is read both as .tile() and as .tileT(). One buffer ` +
                 `cannot be staged in two layouts, so pick one.`);
    }
  }
  return t;
}

/**
 * Which axes are parallel and which are contracted, read from the body.
 *
 * These used to be declared, as `spec.grid` and `spec.reduce`. Declaring them
 * asked the author for something the compiler could already see — the same shape
 * of mistake as a host recomputing `manifest.dispatch` — and it forced a single
 * partition per kernel, which is a matmul-shaped assumption.
 *
 * The rules:
 *
 *   contracted   the axis of a `for (const n of reduce.n)` pass
 *   parallel     an axis used as `at.<name>` in the store's `.tile(...)`
 *
 * "Parallel = the output binding's axes" would have been wrong, and softmax is
 * the counterexample: `y` is `[m, n]` but `n` is reduced, and the body says so by
 * writing `y.tile(at.m, n)` rather than `y.tile(at.m, at.n)`.
 */
function deriveAxisRoles(
  parsed: RowBody,
  axes: readonly AxisIR[],
  node: ts.Node,
): { grid: AxisIR[]; reduce: AxisIR[] } {
  const byName = new Map(axes.map((a) => [a.name, a]));
  const contracted: string[] = [];
  const parallel: string[] = [];
  const push = (xs: string[], n: string) => { if (!xs.includes(n)) xs.push(n); };

  for (const st of parsed.steps) {
    if (st.k === "reduce" || st.k === "store") push(contracted, st.axis);
    if (st.k === "store" || st.k === "storeFrag") {
      for (const c of st.coords) if (c.kind === "at") push(parallel, c.axis);
    }
  }

  for (const n of [...contracted, ...parallel]) {
    if (!byName.has(n)) {
      fail(node, `the body uses axis "${n}", which is not in spec.axes ` +
                 `(${axes.map((a) => a.name).join(", ")})`);
    }
  }
  for (const a of axes) {
    if (!contracted.includes(a.name) && !parallel.includes(a.name)) {
      fail(node, `axis "${a.name}" is declared but the body never uses it, ` +
                 `neither as at.${a.name} in a store nor as a reduce loop`);
    }
  }
  if (parallel.length < 1 || parallel.length > 3) {
    fail(node, `a kernel needs 1-3 parallel axes (WebGPU dispatches in 3 dimensions), ` +
               `and the body's store uses ${parallel.length}`);
  }
  if (contracted.length !== 1) {
    fail(node, `this build supports exactly 1 reduction axis, and the body reduces ` +
               `over ${contracted.length} (${contracted.join(", ") || "none"})`);
  }
  // Reduced and stored along, WITHIN ONE CONTRACTION: `c[m,n] = sum_n …` sums
  // over the axis it also indexes the output by. That is not unimplemented, it is
  // undefined, and it stays an error however far the emitter gets.
  //
  // NOT to be confused with attention, where the head dimension is contracted in
  // `S = Q·Kᵀ` and free in `O = P·V`. That is the same axis in two roles across
  // TWO contractions, which is legal and unimplemented — and it cannot reach this
  // check anyway, because a body with two reduction axes is refused above.
  const both = parallel.filter((n) => contracted.includes(n));
  if (both.length) {
    fail(node, `axis "${both[0]}" is reduced and also stored along, in the same ` +
               `contraction: the sum runs over the axis that indexes the output. ` +
               `Reduce along a different axis, or store along one.`);
  }
  return {
    grid: parallel.map((n) => byName.get(n)!),
    reduce: contracted.map((n) => byName.get(n)!),
  };
}

export function compileToIR(entryFile: string): KernelIR {
  const program = ts.createProgram([entryFile], {
    strict: true,
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    noEmit: true,
    skipLibCheck: true,
  });

  const sf = program.getSourceFile(entryFile);
  if (!sf) fail(undefined, `cannot read ${entryFile}`);

  // Gate 1. If the kernel does not type-check, nothing downstream is meaningful.
  const diags = [
    ...program.getSyntacticDiagnostics(sf),
    ...program.getSemanticDiagnostics(sf),
  ];
  if (diags.length) {
    const text = ts.formatDiagnosticsWithColorAndContext(diags, {
      getCanonicalFileName: (f) => f,
      getCurrentDirectory: ts.sys.getCurrentDirectory,
      getNewLine: () => "\n",
    });
    throw new FrontendError(`the kernel does not type-check:\n\n${text}`);
  }

  const checker = program.getTypeChecker();
  const call = findKernelCall(sf, checker);
  const specNode = call.arguments[0];
  if (!specNode) fail(call, "kernel() needs a spec object");
  const spec = checker.getTypeAtLocation(specNode);

  // ---- name -------------------------------------------------------------
  // `kernel()` declares `readonly name: string`, so unlike tile/grid/reduce/
  // bindings — which are generic and keep their literal types — the name widens
  // and the checker cannot give it back. It is a label, not a shape, so reading
  // the syntax is the right move rather than a workaround.
  const name = readSpecName(specNode);

  // ---- tile -------------------------------------------------------------
  // `tile` is optional. An Axis already carries its own block, so the triple is
  // only a convenience (and the source of the legal-tile diagnostic); when it is
  // absent the blocks come from the axes and the dtype from the bindings.
  const tileT = propType(checker, spec, "tile");
  const declaredTile = tileT
    ? {
        bm: numberLiteral(checker, tileT, "bm", "tile"),
        bn: numberLiteral(checker, tileT, "bn", "tile"),
        bk: numberLiteral(checker, tileT, "bk", "tile"),
      }
    : undefined;

  // ---- axes -------------------------------------------------------------
  // Declared as one list. Which of them are parallel and which are contracted is
  // read from the body by deriveAxisRoles, below.
  const axesT = propType(checker, spec, "axes") ?? fail(specNode, "spec.axes is missing");
  const axes = tupleElements(checker, axesT).map((t) => readAxis(checker, t, "axes"));
  if (!axes.length) fail(specNode, "spec.axes is empty");
  {
    const seen = new Set<string>();
    for (const a of axes) {
      if (seen.has(a.name)) fail(specNode, `spec.axes names "${a.name}" twice`);
      seen.add(a.name);
    }
  }
  // Tile coherence, when a tile is declared. This used to be a tsc error via the
  // kernel() signature, which cost a seven-diagnostic cascade for one mistake and
  // hard-wired "two grid axes, one reduce axis blocked at bk" into the surface.
  // Reported here instead: one error, and kernels that are not matmul-shaped can
  // exist. docs/001 §7 listed suppressing that cascade as outstanding work.
  const byName = new Map(axes.map((a) => [a.name, a]));

  // ---- bindings ---------------------------------------------------------
  const dtype: DTypeName = declaredTile
    ? readDType(checker, propType(checker, tileT!, "dtype")!, "tile")
    : (() => {
        const bt = tupleElements(checker, propType(checker, spec, "bindings")
          ?? fail(specNode, "spec.bindings is missing"))[0];
        return readDType(checker, propType(checker, bt, "dtype")!, "binding 0");
      })();

  const bindingsT = propType(checker, spec, "bindings") ?? fail(specNode, "spec.bindings is missing");
  const bindings: BindingIR[] = tupleElements(checker, bindingsT).map((bt, i) => {
    const bname = stringLiteral(checker, bt, "name", `binding ${i}`);
    const axesT = propType(checker, bt, "axes") ?? fail(specNode, `binding "${bname}" has no axes`);
    const axes = tupleElements(checker, axesT).map((t) => readAxis(checker, t, `binding "${bname}"`));
    if (axes.length !== 2) fail(specNode, `binding "${bname}" must have exactly 2 axes`);

    for (const a of axes) {
      const known = byName.get(a.name);
      if (!known) fail(specNode, `binding "${bname}" uses axis "${a.name}", which is not in spec.axes`);
      if (known.extent !== a.extent) {
        fail(specNode, `binding "${bname}": axis "${a.name}" has extent ${a.extent}, but it is ${known.extent} elsewhere`);
      }
    }

    const bdT = propType(checker, bt, "dtype") ?? fail(specNode, `binding "${bname}" has no dtype`);
    const bdtype = readDType(checker, bdT, `binding "${bname}"`);
    if (bdtype !== dtype) fail(specNode, `binding "${bname}" is ${bdtype} but the tile is ${dtype}`);

    const mode = stringLiteral(checker, bt, "mode", `binding "${bname}"`);
    if (mode !== "read" && mode !== "write") fail(specNode, `binding "${bname}" has mode '${mode}'`);

    return {
      name: bname,
      axes: [axes[0].name, axes[1].name] as const,
      dtype: bdtype,
      mode,
      elements: axes[0].extent * axes[1].extent,
    };
  });

  if (bindings.length > 8) {
    fail(specNode, `${bindings.length} storage buffers exceeds WebGPU's maxStorageBuffersPerShaderStage of 8`);
  }
  if (!bindings.some((b) => b.mode === "write")) fail(specNode, "no output binding");

  // ---- masks -------------------------------------------------------------
  // Derived, never declared. A load or store through a ragged axis is masked;
  // that is the entire rule. The surface's job was to make the user name the
  // identity element, which is the one fact the compiler cannot infer. Deciding
  // WHERE masks go is the compiler's job, and is the demo the project exists for:
  // change one literal and every boundary condition is re-derived.
  const raggedNames = new Set(axes.filter((a) => a.fit === "ragged").map((a) => a.name));

  const maskedLoads: string[] = [];
  for (const b of bindings) {
    for (const ax of b.axes) if (raggedNames.has(ax)) maskedLoads.push(`${b.name}:${ax}`);
  }

  const { padded, schedule, rowBody } = checkCanonicalBody(call, bindings, raggedNames);
  const { grid, reduce } = deriveAxisRoles(rowBody, axes, specNode);

  // `.tileT()` is a property of how the BODY reads a binding, so it is filled in
  // after parsing rather than read from the spec.
  const transposed = collectTransposed(rowBody, specNode);
  for (let i = 0; i < bindings.length; i++) {
    if (transposed.has(bindings[i].name)) bindings[i] = { ...bindings[i], transposed: true };
  }

  // Tile coherence, when a tile is declared. This used to be a tsc error via the
  // kernel() signature, which cost a seven-diagnostic cascade for one mistake and
  // hard-wired "two grid axes, one reduce axis blocked at bk" into the surface.
  // Reported here instead: one error, and kernels that are not matmul-shaped can
  // exist. docs/001 §7 listed suppressing that cascade as outstanding work.
  if (declaredTile && grid.length === 2) {
    const named: [string, number, number][] = [
      [grid[0].name, grid[0].block, declaredTile.bm],
      [grid[1].name, grid[1].block, declaredTile.bn],
      [reduce[0].name, reduce[0].block, declaredTile.bk],
    ];
    for (const [nm, got, want] of named) {
      if (got !== want) {
        fail(specNode, `TSA0051: axis "${nm}" is blocked at ${got}, but the tile says ${want}`);
      }
    }
  }

  // Every read binding touching a ragged axis needs its identity named; the
  // surface enforces this too, but this message names the binding directly.
  for (const b of bindings) {
    const needsPad = b.mode === "read" && b.axes.some((ax) => raggedNames.has(ax));
    if (needsPad && !padded.has(b.name)) {
      fail(specNode,
        `binding "${b.name}" loads through ragged axis ` +
        `"${b.axes.find((ax) => raggedNames.has(ax))}", so out-of-range lanes need an ` +
        `identity: write ${b.name}.tile(...).pad(0).`);
    }
    if (!needsPad && padded.has(b.name)) {
      fail(specNode,
        `binding "${b.name}" has .pad() but no ragged axis — the mask would be dead. ` +
        `Remove it.`);
    }
  }

  // The identity is the BINDING's, not the kernel's. A kernel legitimately wants
  // two — attention pads Q·Kᵀ's operands with `zero` so a masked lane annihilates
  // in the sum, and pads a row max with `negInf` because zero would bias the
  // maximum toward itself.
  for (let i = 0; i < bindings.length; i++) {
    const p = padded.get(bindings[i].name);
    if (p) bindings[i] = { ...bindings[i], pad: p };
  }

  // The MLIR backend still carries one per kernel, and handles one kernel shape.
  // Where a body names two, that backend cannot express it and says so rather
  // than emitting the wrong one.
  const names = new Set(padded.values());
  const pad: PadName = (names.values().next().value ?? "zero") as PadName;
  const multiPad = names.size > 1;

  // The block sizes the emitter needs, whether or not a tile was declared.
  // matmul blocks two parallel axes and one reduction; softmax blocks one
  // parallel axis and reduces along the axis it also stores along.
  const tile = declaredTile ?? (schedule === "matmul"
    ? { bm: grid[0].block, bn: grid[1].block, bk: reduce[0].block }
    : { bm: grid[0].block, bn: reduce[0].block, bk: reduce[0].block });

  return {
    name, schedule, dtype, tile, grid, reduce, bindings, maskedLoads, pad, multiPad,
    body: rowBody,
    ...derive(tile, dtype, grid, undefined, schedule),
  };
}
