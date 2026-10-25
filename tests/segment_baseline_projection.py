"""Project only the exact reviewed timeline-segment additions out of frozen old gates.

Every audited opt-in feature here carries its own projection module so that a frozen
regression sees the tree as it looked before that one feature and nothing else. This one
covers the H3 timeline segmentation: two appended schema inputs, three appended string
ports, the module-level splitter helper and its import, the two execute keyword defaults,
the two derived-value assignments, and the frontend array entries those controls add.
"""
import ast
import copy

INPUT_NAMES = ("segment_count", "segment_split_mode")
OUTPUT_NAMES = ("h3_segments_json", "h3_segments_text", "h3_segments_preset")
HELPER = "_h3_segment_outputs"
MODULE = "h3_segments"

SERIALIZED_ENTRY = '    "segment_count",\n    "segment_split_mode",\n'
DEFAULTS_ENTRY = '    segment_count: 1,\n    segment_split_mode: "\u5e73\u5747\u5206\u6bb5\uff08\u6309\u5206\u6bb5\u6570\uff09 / Even",\n'
EXTENDED_LAYOUT = "[22, 31, 35, 36, 38, SERIALIZED_WIDGET_NAMES.length],"
PUBLISHED_LAYOUT = "[22, 31, 35, 36, SERIALIZED_WIDGET_NAMES.length],"


def without_segment_inputs(inputs):
    kept = [item for item in inputs if item.id not in INPUT_NAMES]
    if len(inputs) - len(kept) != 2:
        raise AssertionError("Expected exactly two appended segment inputs")
    return kept


def without_segment_outputs(outputs):
    kept = [item for item in outputs if item.display_name not in OUTPUT_NAMES]
    if len(outputs) - len(kept) != 3:
        raise AssertionError("Expected exactly three appended segment ports")
    return kept


def without_segment_signature(signature):
    parameters = [item for item in signature.parameters.values() if item.name not in INPUT_NAMES]
    if len(signature.parameters) - len(parameters) != 2:
        raise AssertionError("Expected exactly two appended segment parameters")
    return signature.replace(parameters=parameters)


def project_segment_text(text):
    """Undo the frontend segment entries in a newline-normalized enhancer script."""
    normalized = text.replace("\r\n", "\n")
    for entry in (SERIALIZED_ENTRY, DEFAULTS_ENTRY):
        if normalized.count(entry) != 1:
            raise AssertionError("Expected exactly one segment entry in the widget arrays")
        normalized = normalized.replace(entry, "")
    if normalized.count(EXTENDED_LAYOUT) != 2:
        raise AssertionError("Expected the accepted-layout lists to carry the historical 38 values")
    return normalized.replace(EXTENDED_LAYOUT, PUBLISHED_LAYOUT)


def project_segment_bytes(raw):
    return project_segment_text(raw.decode("utf-8")).encode("utf-8")


def _belongs_to_segments(node):
    if not isinstance(node, ast.Call):
        return False
    first = node.args[0] if node.args else None
    if isinstance(first, ast.Constant) and first.value in INPUT_NAMES:
        return True
    return any(keyword.arg == "display_name" and isinstance(keyword.value, ast.Constant)
               and keyword.value.value in OUTPUT_NAMES for keyword in node.keywords)


def _is_segment_import(node):
    return isinstance(node, ast.ImportFrom) and node.module == MODULE


def _drop_star_arguments(call):
    """Return the number of segment splices removed from one NodeOutput call."""
    kept = []
    removed = 0
    for argument in call.args:
        value = argument.value if isinstance(argument, ast.Starred) else None
        if (isinstance(value, ast.Name) and value.id == "segments") or (
                isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == HELPER):
            removed += 1
            continue
        kept.append(argument)
    call.args = kept
    return removed


def project_segment_ast(tree):
    tree = copy.deepcopy(tree)
    counts = {"import": 0, "helper": 0, "parameters": 0, "inputs": 0, "outputs": 0,
              "assignments": 0, "splices": 0, "inlined": 0}

    class Projection(ast.NodeTransformer):
        def visit_Try(self, node):
            self.generic_visit(node)
            if (len(node.body) == 1 and _is_segment_import(node.body[0]) and node.body[0].level == 1
                    and len(node.handlers) == 1
                    and ast.unparse(node.handlers[0].type) == "ImportError"
                    and len(node.handlers[0].body) == 1
                    and _is_segment_import(node.handlers[0].body[0])
                    and node.handlers[0].body[0].level == 0):
                counts["import"] += 1
                return None
            body = node.body
            position = next((index for index, statement in enumerate(body)
                            if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                            and isinstance(statement.targets[0], ast.Name)
                            and statement.targets[0].id == "restored"), None)
            if position is None:
                return node
            following = body[position + 1] if position + 1 < len(body) else None
            returned = following.value if isinstance(following, ast.Return) else None
            if (not isinstance(returned, ast.Call) or not ast.unparse(returned.func).endswith("NodeOutput")
                    or len(returned.args) != 1 or not isinstance(returned.args[0], ast.Starred)
                    or not isinstance(returned.args[0].value, ast.Name)
                    or returned.args[0].value.id != "restored"):
                raise AssertionError("The reviewed restore path must publish the recovered tuple it keeps locally")
            body[position] = ast.Return(value=ast.Call(
                func=returned.func, args=[ast.Starred(value=body[position].value, ctx=ast.Load())], keywords=[]))
            del body[position + 1]
            counts["inlined"] += 1
            return node

        def visit_FunctionDef(self, node):
            if node.name == HELPER:
                counts["helper"] += 1
                return None
            for field, defaults in (("args", "defaults"), ("kwonlyargs", "kw_defaults")):
                arguments, values = getattr(node.args, field), getattr(node.args, defaults)
                dropped = [index for index, item in enumerate(arguments) if item.arg in INPUT_NAMES]
                if not dropped:
                    continue
                offset = len(arguments) - len(values)
                if any(index < offset for index in dropped):
                    raise AssertionError("Segment parameters must carry keyword defaults")
                for index in reversed(dropped):
                    del arguments[index]
                    if index >= offset:
                        del values[index - offset]
                counts["parameters"] += len(dropped)
            self.generic_visit(node)
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"segments", "restored_duration"}:
                counts["assignments"] += 1
                return None
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            for keyword in node.keywords:
                if keyword.arg in {"inputs", "outputs"} and isinstance(keyword.value, ast.List):
                    kept = [item for item in keyword.value.elts if not _belongs_to_segments(item)]
                    removed = len(keyword.value.elts) - len(kept)
                    counts["inputs" if keyword.arg == "inputs" else "outputs"] += removed
                    keyword.value.elts = kept
            if ast.unparse(node.func).endswith("NodeOutput"):
                counts["splices"] += _drop_star_arguments(node)
            for keyword in node.keywords:
                if keyword.arg in INPUT_NAMES:
                    raise AssertionError("Segment values must be passed positionally")
            return node

    tree = Projection().visit(tree)
    present = {"import": 1, "helper": 1, "parameters": 2, "inputs": 2, "outputs": 3,
               "assignments": 2, "splices": 3, "inlined": 1}
    absent = {name: 0 for name in present}
    if counts not in (present, absent):
        raise AssertionError(f"Unexpected segmentation footprint in the frozen module: {counts}")
    return tree


def project_segment_source(source):
    return ast.unparse(project_segment_ast(ast.parse(source)))
