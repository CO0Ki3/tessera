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
import { derive, type AxisIR, type BindingIR, type DTypeName, type KernelIR } from "./ir.ts";

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
  reduceAxes: readonly AxisIR[],
  bindings: readonly BindingIR[],
  raggedNames: ReadonlySet<string>,
): { padded: Set<string> } {
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

  let forOf = 0, mma = 0, store = 0, zeros = 0;
  const padded = new Set<string>();

  // Tokens (punctuation, keywords, identifiers, literals) carry no admission
  // decision of their own — the node that owns them is already constrained, and
  // a token has no children to recurse into. Two are hazards in their own right.
  const DENIED_TOKENS = new Set<ts.SyntaxKind>([
    ts.SyntaxKind.ThisKeyword,
    ts.SyntaxKind.SuperKeyword,
    ts.SyntaxKind.ImportKeyword,
  ]);

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
        `This build emits only the canonical accumulate-and-store shape; rather than ` +
        `silently compiling something that is not what you wrote, it refuses.`);
    }
    if (ts.isBinaryExpression(n) && n.operatorToken.kind !== ts.SyntaxKind.EqualsToken) {
      fail(n,
        `TSA0016: only whole-expression assignment is admitted in a kernel body, ` +
        `not '${ts.tokenToString(n.operatorToken.kind)}'.`);
    }

    if (ts.isForOfStatement(n)) forOf++;
    else if (ts.isCallExpression(n) && ts.isIdentifier(n.expression)) {
      if (n.expression.text === "mma") mma++;
      if (n.expression.text === "zeros") zeros++;
    } else if (ts.isCallExpression(n) && ts.isPropertyAccessExpression(n.expression)) {
      const method = n.expression.name.text;
      if (method === "store") store++;
      if (method === "pad") {
        // `a.tile(...).pad(0)` — the receiver is the tile() call, whose own
        // receiver names the binding.
        const inner = n.expression.expression;
        const owner = ts.isCallExpression(inner) && ts.isPropertyAccessExpression(inner.expression)
          && ts.isIdentifier(inner.expression.expression)
          ? inner.expression.expression.text : undefined;
        if (!owner || !bindings.some((b) => b.name === owner)) {
          fail(n, `.pad() must be called on a binding's tile, as <binding>.tile(...).pad(0)`);
        }
        const arg = n.arguments[0];
        if (!arg || !ts.isNumericLiteral(arg) || arg.text !== "0") {
          fail(arg ?? n,
            `.pad() takes the reduction's identity element, which for a sum is 0. ` +
            `A non-annihilating pad silently corrupts ragged edges.`);
        }
        padded.add(owner);
      }
    }

    ts.forEachChild(n, visit);
  };
  visit(body.body);

  const shape = `${zeros} zeros / ${forOf} loops / ${mma} mma / ${store} store`;
  if (zeros !== 1 || forOf !== reduceAxes.length || mma !== 1 || store !== 1) {
    fail(body,
      `this build only emits the canonical accumulate-and-store body ` +
      `(1 zeros, ${reduceAxes.length} reduce loop(s), 1 mma, 1 store); this body is ${shape}. ` +
      `General body compilation is not implemented — refusing rather than emitting something ` +
      `that is not what you wrote.`);
  }
  return { padded };
}

// ---------------------------------------------------------------------------

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
  const tileT = propType(checker, spec, "tile") ?? fail(specNode, "spec.tile is missing");
  const tile = {
    bm: numberLiteral(checker, tileT, "bm", "tile"),
    bn: numberLiteral(checker, tileT, "bn", "tile"),
    bk: numberLiteral(checker, tileT, "bk", "tile"),
  };
  const dtypeT = propType(checker, tileT, "dtype") ?? fail(specNode, "tile.dtype is missing");
  const dtype = readDType(checker, dtypeT, "tile");

  // ---- axes -------------------------------------------------------------
  const gridT = propType(checker, spec, "grid") ?? fail(specNode, "spec.grid is missing");
  const grid = tupleElements(checker, gridT).map((t) => readAxis(checker, t, "grid"));
  if (grid.length !== 2) fail(specNode, `spec.grid must have exactly 2 axes, got ${grid.length}`);

  const reduceT = propType(checker, spec, "reduce") ?? fail(specNode, "spec.reduce is missing");
  const reduce = tupleElements(checker, reduceT).map((t) => readAxis(checker, t, "reduce"));
  if (reduce.length !== 1) {
    fail(specNode, `this build supports exactly 1 reduce axis, got ${reduce.length}`);
  }

  // Tile/axis coherence — the surface types enforce this too, but the message
  // here names the mismatch directly instead of cascading.
  if (grid[0].block !== tile.bm) fail(specNode, `grid axis "${grid[0].name}" is blocked at ${grid[0].block}, tile.bm is ${tile.bm}`);
  if (grid[1].block !== tile.bn) fail(specNode, `grid axis "${grid[1].name}" is blocked at ${grid[1].block}, tile.bn is ${tile.bn}`);
  if (reduce[0].block !== tile.bk) fail(specNode, `reduce axis "${reduce[0].name}" is blocked at ${reduce[0].block}, tile.bk is ${tile.bk}`);

  const byName = new Map([...grid, ...reduce].map((a) => [a.name, a]));

  // ---- bindings ---------------------------------------------------------
  const bindingsT = propType(checker, spec, "bindings") ?? fail(specNode, "spec.bindings is missing");
  const bindings: BindingIR[] = tupleElements(checker, bindingsT).map((bt, i) => {
    const bname = stringLiteral(checker, bt, "name", `binding ${i}`);
    const axesT = propType(checker, bt, "axes") ?? fail(specNode, `binding "${bname}" has no axes`);
    const axes = tupleElements(checker, axesT).map((t) => readAxis(checker, t, `binding "${bname}"`));
    if (axes.length !== 2) fail(specNode, `binding "${bname}" must have exactly 2 axes`);

    for (const a of axes) {
      const known = byName.get(a.name);
      if (!known) fail(specNode, `binding "${bname}" uses axis "${a.name}", which is not in grid or reduce`);
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
  const raggedNames = new Set(
    [...grid, ...reduce].filter((a) => a.fit === "ragged").map((a) => a.name));

  const maskedLoads: string[] = [];
  for (const b of bindings) {
    for (const ax of b.axes) if (raggedNames.has(ax)) maskedLoads.push(`${b.name}:${ax}`);
  }

  const { padded } = checkCanonicalBody(call, reduce, bindings, raggedNames);

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

  const pad = 0;

  return {
    name, dtype, tile, grid, reduce, bindings, maskedLoads, pad,
    ...derive(tile, dtype, grid),
  };
}
