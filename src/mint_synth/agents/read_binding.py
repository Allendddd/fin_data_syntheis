"""Recognize bounded native read forms without executing or interpreting shell code."""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path
from typing import Any, Mapping


def _python_reads(code: str) -> list[str]:
    """Only print expressions whose value is actually obtained by a file read."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    expressions = []
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom):
            if (statement.module != "pathlib" or any(name.name != "Path" or name.asname
                                                    for name in statement.names)):
                return []
        elif isinstance(statement, ast.Import):
            if any(name.name != "pathlib" or name.asname for name in statement.names):
                return []
        elif isinstance(statement, ast.Expr):
            expressions.append(statement.value)
        else:
            return []
    if len(expressions) != 1:
        return []
    output = expressions[0]
    if not (isinstance(output, ast.Call) and isinstance(output.func, ast.Name)
            and output.func.id == "print" and len(output.args) == 1 and not output.keywords):
        return []

    def read_value(node):
        if isinstance(node, ast.Subscript):
            # A literal line/character slice is still backed by the same read.
            if any(isinstance(child, (ast.Call, ast.Name)) for child in ast.walk(node.slice)):
                return []
            return read_value(node.value)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return []
        receiver, method = node.func.value, node.func.attr
        if method == "splitlines" and not node.args and not node.keywords:
            return read_value(receiver)
        if (method == "join" and isinstance(receiver, ast.Constant) and receiver.value in {"", "\n"}
                and len(node.args) == 1 and not node.keywords):
            return read_value(node.args[0])
        if method not in {"read", "read_text"} or not isinstance(receiver, ast.Call):
            return []
        creator = receiver.func.id if isinstance(receiver.func, ast.Name) else (
            "Path" if isinstance(receiver.func, ast.Attribute) and receiver.func.attr == "Path"
            and isinstance(receiver.func.value, ast.Name) and receiver.func.value.id == "pathlib" else "")
        if creator != ("open" if method == "read" else "Path") or not receiver.args:
            return []
        if len(receiver.args) > (2 if creator == "open" else 1):
            return []
        path = receiver.args[0]
        if not isinstance(path, ast.Constant) or not isinstance(path.value, str):
            return []
        if len(receiver.args) > 1 and getattr(receiver.args[1], "value", None) not in {"r", "rt"}:
            return []
        if any(keyword.arg == "mode" and getattr(keyword.value, "value", None) not in {"r", "rt"}
               for keyword in receiver.keywords):
            return []
        if any(keyword.arg not in {"mode", "encoding", "errors", "newline"}
               or not isinstance(keyword.value, ast.Constant) for keyword in receiver.keywords):
            return []
        if any(not isinstance(value, ast.Constant) for value in node.args):
            return []
        if any(keyword.arg not in {"encoding", "errors", "size"}
               or not isinstance(keyword.value, ast.Constant) for keyword in node.keywords):
            return []
        return [path.value]

    return read_value(output.args[0])


def source_paths(use: Mapping[str, Any], root: Path) -> list[Path]:
    """Resolve explicit files for readers, cat/head/tail/sed/rg/grep and simple Python reads.

    Shell expansions, fallbacks and arbitrary programs remain unsupported. Actual
    output must independently match source text before any binding is credited.
    """
    args = use.get("input", {})
    if not isinstance(args, Mapping):
        return []
    cwd = Path(args.get("workdir") or args.get("cwd") or root)
    if not cwd.is_absolute():
        cwd = root / cwd
    name = use.get("name")
    if name in {"Read", "file_read"}:
        path = args.get("file_path", args.get("path"))
        return [(cwd / path).resolve()] if isinstance(path, str) and args.get("mode", "view") == "view" else []
    if name not in {"Bash", "exec_command", "shell_command", "shell", "shell_call"}:
        return []
    command = args.get("cmd", args.get("command", args.get("commands")))
    if isinstance(command, list):
        if len(command) == 1:
            command = command[0]
        elif len(command) == 3 and Path(command[0]).name in {"bash", "sh", "zsh"} and command[1] in {"-c", "-lc"}:
            command = command[2]
        elif command and command[0] == "cat":
            command = shlex.join(command)
        else:
            return []
    if not isinstance(command, str) or any(char in command for char in "$`*?~\n\r"):
        return []
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split, lexer.commenters = True, ""
        tokens = list(lexer)
    except ValueError:
        return []
    segments, current = [], []
    for token in tokens:
        if token in {"&&", "|"}:
            segments.append(current)
            current = []
        elif token in {";", "||", "&", ">", ">>", "<", "(", ")"}:
            return []
        else:
            current.append(token)
    segments.append(current)
    paths = []
    for parts in segments:
        if not parts:
            return []
        executable = Path(parts[0]).name
        if executable == "cd" and len(parts) == 2:
            cwd = (cwd / parts[1]).resolve()
            continue
        if executable in {"python", "python3"} and len(parts) == 3 and parts[1] == "-c":
            files = _python_reads(parts[2])
        elif executable in {"cat", "head", "tail", "sed", "rg", "grep"}:
            files, i, pattern = [], 1, executable in {"sed", "rg", "grep"}
            while i < len(parts):
                word = parts[i]
                if word in {"-n", "-c", "--lines", "--bytes", "-m", "--max-count", "-e", "--regexp"}:
                    if executable in {"cat", "rg", "grep"} and word == "-n":
                        i += 1
                        continue
                    if word in {"-e", "--regexp"}:
                        pattern = False
                    elif executable == "sed" and word == "-n":
                        i += 1
                        continue
                    i += 2
                    continue
                if word.startswith("-"):
                    i += 1
                    continue
                if pattern:
                    if executable == "sed" and not re.fullmatch(r"\d+(?:,\d+)?p", word):
                        return []
                    pattern = False
                else:
                    files.append(word)
                i += 1
        else:
            return []
        paths.extend((cwd / path).resolve() for path in files)
    return list(dict.fromkeys(paths))


def observed_source_text(output: str, source: str) -> str:
    """Credit only returned source spans, removing known reader line decorations."""
    if output and output in source:
        return output
    lines = []
    for line in output.splitlines():
        clean = re.sub(r"^\s*\d+(?:→|\t|\s*\|\s|:)\s?", "", line)
        if clean.strip() and clean in source:
            lines.append(clean)
            continue
        # rg/grep with file and line prefixes. The path was separately bound.
        match = re.match(r"^.*?:\d+:(.*)$", line)
        if match and match[1].strip() and match[1] in source:
            lines.append(match[1])
        elif ":" in line:
            for position, char in enumerate(line):
                if char == ":" and line[position + 1:].strip() and line[position + 1:] in source:
                    lines.append(line[position + 1:])
                    break
    return "\n".join(lines)
