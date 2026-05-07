from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

import pandas as pd
import requests
import yaml
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Frame
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"


TREE_SNAPSHOT_JS = r"""
(cfg) => {
  const tree = document.querySelector(cfg.tree);
  if (!tree) {
    return { ok: false, error: `找不到设备树: ${cfg.tree}` };
  }

  const direct = (el, selector) => {
    for (const child of el.children) {
      if (child.matches(selector)) return child;
    }
    return null;
  };

  const labelOf = (node) => {
    const content = direct(node, cfg.tree_content) || node;
    const labelEl = content.querySelector(cfg.tree_label);
    const raw = labelEl ? labelEl.textContent : content.textContent;
    return (raw || "").replace(/\s+/g, " ").trim();
  };

  const hasCheckbox = (node) => {
    const content = direct(node, cfg.tree_content) || node;
    return Boolean(content.querySelector(cfg.tree_checkbox));
  };

  const readNode = (node, path, indexPath) => {
    const label = labelOf(node);
    const childrenBox = direct(node, cfg.tree_children);
    const childNodes = childrenBox
      ? Array.from(childrenBox.children).filter((child) => child.matches(cfg.tree_node))
      : [];
    const itemPath = path.concat(label);
    const item = {
      label,
      path: itemPath,
      pathText: itemPath.join("-"),
      indexPath,
      level: itemPath.length,
      hasCheckbox: hasCheckbox(node),
      isLeaf: childNodes.length === 0,
      children: []
    };
    item.children = childNodes.map((child, index) =>
      readNode(child, itemPath, indexPath.concat(index))
    );
    return item;
  };

  const roots = Array.from(tree.children).filter((child) => child.matches(cfg.tree_node));
  return { ok: true, nodes: roots.map((node, index) => readNode(node, [], [index])) };
}
"""


EXPAND_TREE_JS = r"""
(cfg) => {
  const tree = document.querySelector(cfg.tree);
  if (!tree) return { clicked: 0, missing: true };
  const icons = Array.from(tree.querySelectorAll(cfg.expand_icon));
  let clicked = 0;
  for (const icon of icons) {
    const cls = icon.className || "";
    const node = icon.closest(cfg.tree_node);
    const expandedByNode = node && node.getAttribute("aria-expanded") === "true";
    const expandedByIcon = String(cls).includes("expanded");
    const isLeaf = String(cls).includes("is-leaf");
    if (!isLeaf && !expandedByNode && !expandedByIcon) {
      icon.click();
      clicked += 1;
    }
  }
  return { clicked, missing: false };
}
"""


CLEAR_TREE_CHECKS_JS = r"""
(cfg) => {
  const tree = document.querySelector(cfg.tree);
  if (!tree) return { clicked: 0, missing: true };
  const checkedSelector = `${cfg.tree_checkbox}.${cfg.checked_class}`;
  const boxes = Array.from(tree.querySelectorAll(checkedSelector));
  let clicked = 0;
  for (const box of boxes) {
    box.click();
    clicked += 1;
  }
  return { clicked, missing: false };
}
"""


CHECK_NODE_BY_PATH_JS = r"""
({ cfg, path }) => {
  const tree = document.querySelector(cfg.tree);
  if (!tree) return { ok: false, error: `找不到设备树: ${cfg.tree}` };

  const direct = (el, selector) => {
    for (const child of el.children) {
      if (child.matches(selector)) return child;
    }
    return null;
  };

  const labelOf = (node) => {
    const content = direct(node, cfg.tree_content) || node;
    const labelEl = content.querySelector(cfg.tree_label);
    const raw = labelEl ? labelEl.textContent : content.textContent;
    return (raw || "").replace(/\s+/g, " ").trim();
  };

  const roots = Array.from(tree.children).filter((child) => child.matches(cfg.tree_node));

  const find = (nodes, depth) => {
    for (const node of nodes) {
      if (labelOf(node) !== path[depth]) continue;
      if (depth === path.length - 1) return node;
      const childrenBox = direct(node, cfg.tree_children);
      const childNodes = childrenBox
        ? Array.from(childrenBox.children).filter((child) => child.matches(cfg.tree_node))
        : [];
      const matched = find(childNodes, depth + 1);
      if (matched) return matched;
    }
    return null;
  };

  const node = find(roots, 0);
  if (!node) return { ok: false, error: `找不到节点: ${path.join("-")}` };
  const content = direct(node, cfg.tree_content) || node;
  const checkbox = content.querySelector(cfg.tree_checkbox);
  if (!checkbox) return { ok: false, error: `节点没有复选框: ${path.join("-")}` };
  checkbox.scrollIntoView({ block: "center", inline: "nearest" });
  const cls = checkbox.className || "";
  const checked = String(cls).includes(cfg.checked_class) || checkbox.getAttribute("aria-checked") === "true";
  if (!checked) checkbox.click();
  return { ok: true };
}
"""


INSPECT_JS = r"""
() => {
  const text = (el) => (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
  const inputs = Array.from(document.querySelectorAll("input")).slice(0, 30).map((el, i) => ({
    index: i,
    placeholder: el.getAttribute("placeholder") || "",
    value: el.value || "",
    class: el.className || ""
  }));
  const buttons = Array.from(document.querySelectorAll("button, .el-button, [role='button']"))
    .map(text).filter(Boolean).slice(0, 80);
  const trees = Array.from(document.querySelectorAll(".el-tree, [role='tree'], [class*='tree'], [class*='Tree'], .ztree")).map((el, i) => ({
    index: i,
    class: el.className || "",
    text: text(el).slice(0, 200)
  }));
  const iframes = Array.from(document.querySelectorAll("iframe")).map((el, i) => ({
    index: i,
    src: el.getAttribute("src") || "",
    name: el.getAttribute("name") || "",
    id: el.id || "",
    class: el.className || ""
  }));
  return { title: document.title, url: location.href, inputs, buttons, trees, iframes };
}
"""


MATCHING_IFRAME_JS = r"""
(containsText) => {
  const frames = Array.from(document.querySelectorAll("iframe"));
  const matched = frames.find((el) => {
    const src = el.getAttribute("src") || "";
    const name = el.getAttribute("name") || "";
    return src.includes(containsText) || name.includes(containsText);
  });
  if (!matched) return null;
  return {
    src: matched.getAttribute("src") || "",
    name: matched.getAttribute("name") || "",
    id: matched.id || ""
  };
}
"""


TREE_EXISTS_JS = r"""
(cfg) => Boolean(document.querySelector(cfg.tree))
"""


ZTREE_EXISTS_JS = r"""
() => Boolean(document.querySelector(".ztree, ul.ztree"))
"""


MINI_TREE_EXISTS_JS = r"""
() => Boolean(document.querySelector(".mini-tree, .mini-treegrid, [class*='mini-tree']"))
"""


CLICK_TEXT_OR_VALUE_JS = r"""
(targetText) => {
  const norm = (value) => String(value || "").replace(/\s+/g, " ").trim();
  const visible = (el) => {
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
  };
  const preferred = [
    "button",
    "input",
    "a",
    ".mini-button",
    ".mini-button-text",
    ".mini-menuitem",
    ".mini-menuitem-text",
    "[role='button']"
  ].join(",");
  const all = Array.from(document.querySelectorAll("body *")).filter(visible);
  const matched = all.find((el) =>
    norm(el.value) === targetText ||
    norm(el.getAttribute("title")) === targetText ||
    norm(el.innerText || el.textContent) === targetText
  );
  if (!matched) return { ok: false };
  const clickable = matched.closest(preferred) || matched;
  clickable.scrollIntoView({ block: "center", inline: "center" });
  clickable.click();
  return { ok: true, tag: clickable.tagName, className: clickable.className || "" };
}
"""


MINI_EXPAND_JS = r"""
() => {
  const tree = document.querySelector(".mini-tree.treeBar, .mini-tree, .mini-treegrid, [class*='mini-tree']");
  if (!tree) return { clicked: 0, missing: true };
  const collapsed = Array.from(tree.querySelectorAll(".mini-tree-nodetitle.mini-tree-collapse"));
  let clicked = 0;
  for (const title of collapsed) {
    const icon = title.querySelector(".mini-tree-node-ecicon") || title.querySelector(".mini-tree-nodeshow") || title;
    icon.scrollIntoView({ block: "center", inline: "nearest" });
    icon.click();
    clicked += 1;
  }
  return { clicked, missing: false };
}
"""


MINI_SNAPSHOT_JS = r"""
() => {
  const tree = document.querySelector(".mini-tree.treeBar, .mini-tree, .mini-treegrid, [class*='mini-tree']");
  if (!tree) return { ok: false, error: "找不到 MiniUI 设备树: .mini-tree" };

  const norm = (value) => String(value || "").replace(/\s+/g, " ").trim();
  const attrsOf = (el) => {
    const attrs = {};
    if (!el) return attrs;
    for (const attr of Array.from(el.attributes || [])) attrs[attr.name] = attr.value;
    return attrs;
  };
  const flattenMiniData = (items, out = []) => {
    for (const item of Array.isArray(items) ? items : []) {
      out.push(item);
      flattenMiniData(item.children || item._children || item.nodes || [], out);
    }
    return out;
  };
  const rawLabelOf = (item) => norm(
    item.text || item.name || item.equipmentName || item.deviceName || item.label ||
    item.displayName || item.title || item.NAME || item.TEXT || ""
  );
  const mapRawData = (items, parentPath = [], out = {}) => {
    for (const item of Array.isArray(items) ? items : []) {
      const label = rawLabelOf(item);
      const path = label ? parentPath.concat(label) : parentPath;
      if (label) out[path.join("-")] = item;
      mapRawData(item.children || item._children || item.nodes || [], path, out);
    }
    return out;
  };
  let rawNodes = [];
  let rawByPath = {};
  try {
    const control = window.mini && tree.id ? mini.get(tree.id) : null;
    const data = control && control.getData ? control.getData() : (control && control.data ? control.data : []);
    rawNodes = flattenMiniData(data);
    rawByPath = mapRawData(data);
  } catch (err) {
    rawNodes = [];
    rawByPath = {};
  }
  const titles = Array.from(tree.querySelectorAll(".mini-tree-nodetitle"));
  const roots = [];
  const stack = [];

  for (let index = 0; index < titles.length; index += 1) {
    const title = titles[index];
    const labelEl = title.querySelector(".mini-tree-nodetext");
    const showEl = title.querySelector(".mini-tree-nodeshow");
    const checkbox = title.querySelector(".mini-tree-checkbox");
    const row = title.closest("tr");
    const label = norm(labelEl ? labelEl.textContent : title.textContent);
    if (!label) continue;

    const level = title.querySelectorAll(".mini-tree-indent").length + 1;
    const parentPath = level > 1 && stack[level - 2] ? stack[level - 2].path : [];
    const path = parentPath.concat(label);
    const rawNode = rawByPath[path.join("-")] || rawNodes[index] || {};
    const item = {
      label,
      path,
      pathText: path.join("-"),
      indexPath: path,
      level,
      hasCheckbox: Boolean(title.querySelector(".mini-tree-checkbox")),
      isLeaf: !String(title.className || "").includes("mini-tree-parentNode"),
      domId: title.id || "",
      rowId: row ? row.id || "" : "",
      checkboxId: checkbox ? checkbox.id || "" : "",
      raw: rawNode,
      attrs: Object.assign({}, attrsOf(row), attrsOf(title), attrsOf(showEl), attrsOf(labelEl), attrsOf(checkbox)),
      children: []
    };
    item.valueCandidates = [
      rawNode.id,
      rawNode.ID,
      rawNode.deviceId,
      rawNode.devId,
      rawNode.meterId,
      rawNode.nodeId,
      rawNode.code,
      rawNode.value,
      item.domId,
      item.rowId,
      item.checkboxId,
      item.attrs.id,
      item.attrs.uid,
      item.attrs.value,
      item.attrs["data-id"],
      item.attrs["data-value"],
      label,
      item.pathText
    ].filter(Boolean);

    stack[level - 1] = item;
    stack.length = level;
    if (level === 1 || !stack[level - 2]) {
      roots.push(item);
    } else {
      stack[level - 2].children.push(item);
    }
  }

  return { ok: true, nodes: roots };
}
"""


MINI_CLEAR_CHECKS_JS = r"""
() => {
  const tree = document.querySelector(".mini-tree.treeBar, .mini-tree, .mini-treegrid, [class*='mini-tree']");
  if (!tree) return { clicked: 0, missing: true };
  const checked = Array.from(tree.querySelectorAll(".mini-tree-checkbox"))
    .filter((box) => {
      const cls = String(box.className || "").toLowerCase();
      return cls.includes("checked") && !cls.includes("unchecked");
    });
  let clicked = 0;
  for (const box of checked) {
    box.scrollIntoView({ block: "center", inline: "nearest" });
    box.click();
    clicked += 1;
  }
  return { clicked, missing: false };
}
"""


MINI_CHECK_NODE_BY_PATH_JS = r"""
({ path }) => {
  const tree = document.querySelector(".mini-tree.treeBar, .mini-tree, .mini-treegrid, [class*='mini-tree']");
  if (!tree) return { ok: false, error: "找不到 MiniUI 设备树: .mini-tree" };

  const norm = (value) => String(value || "").replace(/\s+/g, " ").trim();
  const titles = Array.from(tree.querySelectorAll(".mini-tree-nodetitle"));
  const stack = [];

  for (const title of titles) {
    const labelEl = title.querySelector(".mini-tree-nodetext");
    const label = norm(labelEl ? labelEl.textContent : title.textContent);
    if (!label) continue;

    const level = title.querySelectorAll(".mini-tree-indent").length + 1;
    const parentPath = level > 1 && stack[level - 2] ? stack[level - 2] : [];
    const currentPath = parentPath.concat(label);
    stack[level - 1] = currentPath;
    stack.length = level;

    if (currentPath.length === path.length && currentPath.every((part, index) => part === path[index])) {
      const checkbox = title.querySelector(".mini-tree-checkbox");
      if (!checkbox) return { ok: false, error: `节点没有复选框: ${path.join("-")}` };
      checkbox.scrollIntoView({ block: "center", inline: "nearest" });
      const cls = String(checkbox.className || "").toLowerCase();
      const alreadyChecked = cls.includes("checked") && !cls.includes("unchecked");
      if (!alreadyChecked) checkbox.click();
      return { ok: true };
    }
  }

  return { ok: false, error: `找不到节点: ${path.join("-")}` };
}
"""


ZTREE_EXPAND_JS = r"""
() => {
  const tree = document.querySelector(".ztree, ul.ztree");
  if (!tree) return { clicked: 0, missing: true };
  const switches = Array.from(tree.querySelectorAll("span.switch, span.button.switch"));
  let clicked = 0;
  for (const icon of switches) {
    const cls = String(icon.className || "");
    if (cls.includes("close") && !cls.includes("noline_docu")) {
      icon.click();
      clicked += 1;
    }
  }
  return { clicked, missing: false };
}
"""


ZTREE_SNAPSHOT_JS = r"""
() => {
  const tree = document.querySelector(".ztree, ul.ztree");
  if (!tree) return { ok: false, error: "找不到 zTree 设备树: .ztree" };

  const direct = (el, selector) => {
    for (const child of el.children) {
      if (child.matches(selector)) return child;
    }
    return null;
  };

  const labelOf = (li) => {
    const a = direct(li, "a");
    const labelEl = a ? (a.querySelector(".node_name") || a.querySelector("span[id$='_span']")) : null;
    const raw = labelEl ? labelEl.textContent : (a ? a.textContent : li.textContent);
    return (raw || "").replace(/\s+/g, " ").trim();
  };

  const hasCheckbox = (li) => {
    const a = direct(li, "a");
    return Boolean(a && a.querySelector("span.chk"));
  };

  const readNode = (li, path, indexPath) => {
    const label = labelOf(li);
    const childrenBox = direct(li, "ul");
    const childNodes = childrenBox
      ? Array.from(childrenBox.children).filter((child) => child.tagName === "LI")
      : [];
    const itemPath = path.concat(label);
    const item = {
      label,
      path: itemPath,
      pathText: itemPath.join("-"),
      indexPath,
      level: itemPath.length,
      hasCheckbox: hasCheckbox(li),
      isLeaf: childNodes.length === 0,
      children: []
    };
    item.children = childNodes.map((child, index) =>
      readNode(child, itemPath, indexPath.concat(index))
    );
    return item;
  };

  const roots = Array.from(tree.children).filter((child) => child.tagName === "LI");
  return { ok: true, nodes: roots.map((node, index) => readNode(node, [], [index])) };
}
"""


ZTREE_CLEAR_CHECKS_JS = r"""
() => {
  const tree = document.querySelector(".ztree, ul.ztree");
  if (!tree) return { clicked: 0, missing: true };
  const boxes = Array.from(tree.querySelectorAll("span.chk"))
    .filter((box) => String(box.className || "").includes("checkbox_true_full"));
  let clicked = 0;
  for (const box of boxes) {
    box.click();
    clicked += 1;
  }
  return { clicked, missing: false };
}
"""


ZTREE_CHECK_NODE_BY_PATH_JS = r"""
({ path }) => {
  const tree = document.querySelector(".ztree, ul.ztree");
  if (!tree) return { ok: false, error: "找不到 zTree 设备树: .ztree" };

  const direct = (el, selector) => {
    for (const child of el.children) {
      if (child.matches(selector)) return child;
    }
    return null;
  };

  const labelOf = (li) => {
    const a = direct(li, "a");
    const labelEl = a ? (a.querySelector(".node_name") || a.querySelector("span[id$='_span']")) : null;
    const raw = labelEl ? labelEl.textContent : (a ? a.textContent : li.textContent);
    return (raw || "").replace(/\s+/g, " ").trim();
  };

  const roots = Array.from(tree.children).filter((child) => child.tagName === "LI");

  const find = (nodes, depth) => {
    for (const li of nodes) {
      if (labelOf(li) !== path[depth]) continue;
      if (depth === path.length - 1) return li;
      const childrenBox = direct(li, "ul");
      const childNodes = childrenBox
        ? Array.from(childrenBox.children).filter((child) => child.tagName === "LI")
        : [];
      const matched = find(childNodes, depth + 1);
      if (matched) return matched;
    }
    return null;
  };

  const li = find(roots, 0);
  if (!li) return { ok: false, error: `找不到节点: ${path.join("-")}` };
  const a = direct(li, "a");
  const checkbox = a ? a.querySelector("span.chk") : null;
  if (!checkbox) return { ok: false, error: `节点没有复选框: ${path.join("-")}` };
  checkbox.scrollIntoView({ block: "center", inline: "nearest" });
  const cls = String(checkbox.className || "");
  if (!cls.includes("checkbox_true_full")) checkbox.click();
  return { ok: true };
}
"""


Scope = Page | Frame


@dataclass
class TreeChoice:
    label: str
    path: list[str]
    path_text: str
    leaves: list[dict[str, Any]]
    node: dict[str, Any]


@dataclass
class ExportBatch:
    label: str
    nodes: list[dict[str, Any]]
    leaf_count: int
    uses_leaf_devices: bool


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def strip_wrapping_brackets(value: str) -> str:
    value = value.strip()
    pairs = {
        "[": "]",
        "【": "】",
        "(": ")",
        "（": "）",
    }
    if len(value) >= 2 and value[0] in pairs and value[-1] == pairs[value[0]]:
        return value[1:-1].strip()
    return value


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f"（默认: {default}）" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return strip_wrapping_brackets(value) or (default or "")


def sanitize_filename(value: str, max_length: int = 120) -> str:
    value = re.sub(r'[\\/:*?"<>|]+', "-", value)
    value = re.sub(r"\s+", "_", value).strip("._- ")
    return value[:max_length] or "export"


def sanitize_time_for_filename(value: str) -> str:
    return sanitize_filename(value.replace(":", "-").replace("/", "-"), max_length=40)


def flatten_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for node in nodes:
        result.append(node)
        result.extend(flatten_nodes(node.get("children", [])))
    return result


def leaves_under(node: dict[str, Any]) -> list[dict[str, Any]]:
    children = node.get("children", [])
    if not children:
        return [node] if node.get("hasCheckbox") else []
    leaves: list[dict[str, Any]] = []
    for child in children:
        leaves.extend(leaves_under(child))
    return leaves


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def parse_selection(value: str, count: int) -> list[int]:
    value = value.strip().lower()
    if value in {"all", "a", "全部"}:
        return list(range(count))

    selected: set[int] = set()
    for part in re.split(r"[,，\s]+", value):
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            if start > end:
                start, end = end, start
            selected.update(range(start - 1, end))
        else:
            selected.add(int(part) - 1)

    invalid = [i + 1 for i in selected if i < 0 or i >= count]
    if invalid:
        raise ValueError(f"选择序号超出范围: {invalid}")
    return sorted(selected)


def normalize_interval(user_value: str, cfg: dict[str, Any]) -> tuple[str, str]:
    user_value = strip_wrapping_brackets(user_value).strip().lower()
    for key, item in (cfg.get("intervals") or {}).items():
        aliases = [str(v).lower() for v in item.get("aliases", [])]
        if user_value == key.lower() or user_value in aliases:
            return key, str(item["label"])
    valid = " / ".join(item["label"] for item in (cfg.get("intervals") or {}).values())
    raise ValueError(f"无法识别查询间隔: {user_value}，可选: {valid}")


def ensure_dirs(cfg: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    profile = PROJECT_DIR / cfg.get("browser_profile_dir", "browser_profile")
    downloads = PROJECT_DIR / cfg.get("download_dir", "downloads")
    batches = downloads / "batches"
    output = PROJECT_DIR / cfg.get("output_dir", "output")
    for path in (profile, downloads, batches, output):
        path.mkdir(parents=True, exist_ok=True)
    return profile, downloads, batches, output


def open_or_login(page: Page, cfg: dict[str, Any]) -> None:
    url = cfg.get("website_url") or ask("请输入平台网址")
    if url:
        page.goto(url, wait_until="domcontentloaded")
    print("\n请在打开的浏览器中登录，并进入“冷热量表历史记录”页面。")
    print("如果已经在正确页面，直接回到这里按回车。")
    input("准备好后按回车继续...")


def absolutize_url(page: Page, src: str) -> str:
    return page.evaluate(
        "(src) => new URL(src, location.href).href",
        src,
    )


def open_matching_iframe_as_page(page: Page, cfg: dict[str, Any]) -> None:
    contains_text = cfg.get("app_frame_url_contains")
    target_url = ""
    if contains_text:
        try:
            match = page.evaluate(MATCHING_IFRAME_JS, contains_text)
        except PlaywrightError:
            match = None
        if match and match.get("src"):
            target_url = absolutize_url(page, match["src"])
    if not target_url:
        direct_url = cfg.get("app_direct_url")
        if direct_url and direct_url not in page.url:
            target_url = direct_url
    if not target_url:
        return
    if page.url == target_url:
        return
    print(f"正在进入实际功能页面: {target_url}")
    page.goto(target_url, wait_until="domcontentloaded")
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except PlaywrightTimeoutError:
        pass


def frame_label(scope: Scope) -> str:
    if isinstance(scope, Page):
        return "page"
    name = scope.name or "(no-name)"
    return f"frame name={name!r} url={scope.url!r}"


def find_app_scope_once(page: Page, cfg: dict[str, Any]) -> tuple[Scope, str] | None:
    js_cfg = cfg["selectors"]
    scopes: list[Scope] = [page.main_frame]
    scopes.extend(frame for frame in page.frames if frame is not page.main_frame)

    for scope in scopes:
        try:
            if scope.evaluate(TREE_EXISTS_JS, js_cfg):
                print(f"已找到 Element 设备树所在区域: {frame_label(scope)}")
                return scope, "element"
        except PlaywrightError:
            continue

    for scope in scopes:
        try:
            if scope.evaluate(ZTREE_EXISTS_JS):
                print(f"已找到 zTree 设备树所在区域: {frame_label(scope)}")
                return scope, "ztree"
        except PlaywrightError:
            continue

    for scope in scopes:
        try:
            if scope.evaluate(MINI_TREE_EXISTS_JS):
                print(f"已找到 MiniUI 设备树所在区域: {frame_label(scope)}")
                return scope, "mini"
        except PlaywrightError:
            continue

    return None


def find_app_scope(page: Page, cfg: dict[str, Any]) -> tuple[Scope, str]:
    for _ in range(20):
        found = find_app_scope_once(page, cfg)
        if found:
            return found
        page.wait_for_timeout(500)
    raise RuntimeError("找不到设备树。请确认已进入“冷热量表历史记录”页面，或运行 .\\inspect.bat 查看页面结构。")


def click_text(scope: Scope, text: str, timeout: int = 5000) -> bool:
    candidates = [
        scope.get_by_role("button", name=text, exact=True),
        scope.get_by_text(text, exact=True),
        scope.locator(f"text={text}"),
    ]
    for locator in candidates:
        try:
            locator.first.click(timeout=timeout)
            return True
        except PlaywrightError:
            continue
    try:
        result = scope.evaluate(CLICK_TEXT_OR_VALUE_JS, text)
        return bool(result.get("ok"))
    except PlaywrightError:
        return False
    return False


def expand_tree(scope: Scope, page: Page, cfg: dict[str, Any], tree_kind: str, max_rounds: int = 30) -> None:
    js_cfg = cfg["selectors"]
    for _ in range(max_rounds):
        if tree_kind == "mini":
            result = scope.evaluate(MINI_EXPAND_JS)
        elif tree_kind == "ztree":
            result = scope.evaluate(ZTREE_EXPAND_JS)
        else:
            result = scope.evaluate(EXPAND_TREE_JS, js_cfg)
        if result.get("missing"):
            if tree_kind == "mini":
                raise RuntimeError("找不到设备树: .mini-tree")
            if tree_kind == "ztree":
                raise RuntimeError("找不到设备树: .ztree")
            raise RuntimeError(f"找不到设备树: {js_cfg.get('tree')}")
        if result.get("clicked", 0) == 0:
            return
        page.wait_for_timeout(300)


def read_tree(scope: Scope, page: Page, cfg: dict[str, Any], tree_kind: str) -> list[dict[str, Any]]:
    expand_tree(scope, page, cfg, tree_kind)
    if tree_kind == "ztree":
        result = scope.evaluate(ZTREE_SNAPSHOT_JS)
    elif tree_kind == "mini":
        result = scope.evaluate(MINI_SNAPSHOT_JS)
    else:
        result = scope.evaluate(TREE_SNAPSHOT_JS, cfg["selectors"])
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "读取设备树失败"))
    return result.get("nodes", [])


def build_choices(nodes: list[dict[str, Any]], selection_level: int) -> list[TreeChoice]:
    choices: list[TreeChoice] = []
    for node in flatten_nodes(nodes):
        if node.get("level") != selection_level:
            continue
        leaves = leaves_under(node)
        if not leaves:
            continue
        choices.append(
            TreeChoice(
                label=node["label"],
                path=node["path"],
                path_text=node["pathText"],
                leaves=leaves,
                node=node,
            )
        )
    return choices


def choose_tree_choices(scope: Scope, page: Page, cfg: dict[str, Any], tree_kind: str) -> tuple[str, list[TreeChoice]]:
    nodes = read_tree(scope, page, cfg, tree_kind)
    preferred_levels = list(cfg.get("selection_levels_preferred") or [2, 3])
    if int(cfg.get("selection_level", 2)) not in preferred_levels:
        preferred_levels.insert(0, int(cfg.get("selection_level", 2)))

    choices: list[TreeChoice] = []
    chosen_level = preferred_levels[0]
    for level in preferred_levels:
        choices = build_choices(nodes, int(level))
        if choices:
            chosen_level = int(level)
            break
    if not choices:
        raise RuntimeError("没有找到包含设备的目录，请确认设备树已加载。")

    print(f"\n可选择的设备目录（当前显示第 {chosen_level} 级，优先二级，其次三级）：")
    for index, choice in enumerate(choices, start=1):
        print(f"{index:>3}. {choice.path_text}  ({len(choice.leaves)} 台)")

    while True:
        raw = ask("\n请输入目录序号，支持 1,3,5-8 或 all")
        try:
            indexes = parse_selection(raw, len(choices))
            break
        except (ValueError, TypeError) as exc:
            print(f"输入有误: {exc}")

    selected = [choices[i] for i in indexes]
    device_name = "+".join(choice.path_text for choice in selected)
    return device_name, selected


def make_export_batches(choices: list[TreeChoice], max_devices: int) -> list[ExportBatch]:
    total_leaves = sum(len(choice.leaves) for choice in choices)
    if total_leaves <= max_devices:
        return [
            ExportBatch(
                label="+".join(choice.path_text for choice in choices),
                nodes=[leaf for choice in choices for leaf in choice.leaves],
                leaf_count=total_leaves,
                uses_leaf_devices=False,
            )
        ]

    batches: list[ExportBatch] = []
    for choice in choices:
        if len(choice.leaves) <= max_devices:
            batches.append(
                ExportBatch(
                    label=choice.path_text,
                    nodes=choice.leaves,
                    leaf_count=len(choice.leaves),
                    uses_leaf_devices=False,
                )
            )
        else:
            for index, leaf_batch in enumerate(chunked(choice.leaves, max_devices), start=1):
                batches.append(
                    ExportBatch(
                        label=f"{choice.path_text}_batch{index:03d}",
                        nodes=leaf_batch,
                        leaf_count=len(leaf_batch),
                        uses_leaf_devices=True,
                    )
                )
    return batches


def set_time_inputs(scope: Scope, page: Page, cfg: dict[str, Any], start_time: str, end_time: str) -> None:
    selectors = cfg["selectors"]
    locator = scope.locator(selectors["time_input_selector"])
    if locator.count() <= max(int(selectors["start_time_input_index"]), int(selectors["end_time_input_index"])):
        raise RuntimeError(f"找不到足够的时间输入框: {selectors['time_input_selector']}")

    for index, value in (
        (int(selectors["start_time_input_index"]), start_time),
        (int(selectors["end_time_input_index"]), end_time),
    ):
        field = locator.nth(index)
        field.click()
        page.keyboard.press("Control+A")
        field.fill(value)
        page.keyboard.press("Enter")
        page.wait_for_timeout(200)


def set_interval(scope: Scope, page: Page, interval_label: str) -> None:
    if not click_text(scope, interval_label, timeout=5000):
        raise RuntimeError(f"找不到查询间隔按钮: {interval_label}")
    page.wait_for_timeout(300)


def clear_selected_devices(scope: Scope, page: Page, cfg: dict[str, Any], tree_kind: str) -> None:
    clear_text = cfg.get("buttons", {}).get("clear")
    if clear_text:
        click_text(scope, clear_text, timeout=1500)
        page.wait_for_timeout(300)
    if tree_kind == "ztree":
        scope.evaluate(ZTREE_CLEAR_CHECKS_JS)
    elif tree_kind == "mini":
        scope.evaluate(MINI_CLEAR_CHECKS_JS)
    else:
        scope.evaluate(CLEAR_TREE_CHECKS_JS, cfg["selectors"])
    page.wait_for_timeout(300)


def select_devices(scope: Scope, page: Page, cfg: dict[str, Any], tree_kind: str, devices: list[dict[str, Any]]) -> None:
    clear_selected_devices(scope, page, cfg, tree_kind)
    expand_tree(scope, page, cfg, tree_kind)
    for device in devices:
        if tree_kind == "ztree":
            result = scope.evaluate(ZTREE_CHECK_NODE_BY_PATH_JS, {"path": device["path"]})
        elif tree_kind == "mini":
            result = scope.evaluate(MINI_CHECK_NODE_BY_PATH_JS, {"path": device["path"]})
        else:
            result = scope.evaluate(CHECK_NODE_BY_PATH_JS, {"cfg": cfg["selectors"], "path": device["path"]})
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "选择设备失败"))
    page.wait_for_timeout(500)


def query(scope: Scope, page: Page, cfg: dict[str, Any]) -> None:
    query_text = cfg.get("buttons", {}).get("query", "查询")
    if not click_text(scope, query_text, timeout=8000):
        raise RuntimeError(f"找不到查询按钮: {query_text}")
    wait_seconds = float(cfg.get("wait_after_query_seconds", 2))
    try:
        page.wait_for_load_state("networkidle", timeout=30000)
    except PlaywrightTimeoutError:
        pass
    if wait_seconds > 0:
        page.wait_for_timeout(int(wait_seconds * 1000))


def export_download(scope: Scope, page: Page, cfg: dict[str, Any], batches_dir: Path, batch_index: int) -> Path:
    export_text = cfg.get("buttons", {}).get("export", "导出")
    export_without_chart_text = cfg.get("buttons", {}).get("export_without_chart", "")
    trigger_selector = cfg.get("selectors", {}).get("export_dropdown_trigger", "")
    timeout = int(cfg.get("export_timeout_ms", 120000))

    with page.expect_download(timeout=timeout) as download_info:
        opened_menu = False
        if trigger_selector:
            for i in range(min(page.locator(trigger_selector).count(), 8)):
                try:
                    scope.locator(trigger_selector).nth(i).click(timeout=1000)
                    opened_menu = True
                    page.wait_for_timeout(300)
                    if click_text(scope, export_text, timeout=2000):
                        break
                except PlaywrightError:
                    continue
        if not opened_menu:
            if not click_text(scope, export_text, timeout=8000):
                raise RuntimeError(f"找不到导出按钮: {export_text}")
        if export_without_chart_text:
            page.wait_for_timeout(500)
            if not click_text(scope, export_without_chart_text, timeout=8000):
                raise RuntimeError(f"找不到导出类型按钮: {export_without_chart_text}")

    download = download_info.value
    suggested = sanitize_filename(download.suggested_filename or f"batch_{batch_index}.xlsx")
    suffix = Path(suggested).suffix or ".xlsx"
    target = batches_dir / f"batch_{batch_index:03d}{suffix}"
    download.save_as(target)
    return target


def read_table_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")
    raise ValueError(f"不支持的导出文件格式: {path.name}")


def merge_files(files: list[Path], output_path: Path) -> None:
    frames: list[pd.DataFrame] = []
    for file in files:
        df = read_table_file(file)
        if not df.empty:
            frames.append(df)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output_path.suffix.lower() == ".csv":
        merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        merged.to_excel(output_path, index=False)


def guess_body_format(headers: dict[str, str], post_data: str | None) -> str:
    content_type = ""
    for key, value in headers.items():
        if key.lower() == "content-type":
            content_type = value.lower()
            break
    if "application/json" in content_type:
        return "json"
    if "application/x-www-form-urlencoded" in content_type:
        return "form"
    if post_data:
        stripped = post_data.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return "json"
        if "=" in stripped:
            return "form"
    return "query"


def parse_request_payload(capture: dict[str, Any]) -> dict[str, Any]:
    method = str(capture.get("method", "GET")).upper()
    body_format = capture.get("body_format") or guess_body_format(capture.get("headers", {}), capture.get("post_data"))
    post_data = capture.get("post_data") or ""
    if method == "GET":
        return dict(parse_qsl(urlsplit(capture["url"]).query, keep_blank_values=True))
    if body_format == "json":
        try:
            data = json.loads(post_data) if post_data else {}
            return data if isinstance(data, dict) else {"_json": data}
        except json.JSONDecodeError:
            return {}
    if body_format == "form":
        return dict(parse_qsl(post_data, keep_blank_values=True))
    return {}


def write_capture_file(capture: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(capture, f, allow_unicode=True, sort_keys=False)


def load_capture_file(cfg: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_DIR / cfg.get("api_capture_file", "api_capture.yaml")
    if not path.exists():
        raise RuntimeError(f"找不到接口抓取文件: {path}。请先运行 capture_api.bat。")
    with path.open("r", encoding="utf-8") as f:
        capture = yaml.safe_load(f) or {}
    if is_static_asset(capture):
        raise RuntimeError(f"接口抓取文件指向静态资源: {capture.get('url')}。请重新运行 capture_api.bat。")
    return capture


def header_value(headers: dict[str, str], name: str) -> str:
    lower = name.lower()
    for key, value in headers.items():
        if key.lower() == lower:
            return value
    return ""


def response_filename(headers: dict[str, str], fallback: str, extension: str) -> str:
    disposition = header_value(headers, "content-disposition")
    match = re.search(r'filename\*=UTF-8\'\'([^;]+)', disposition, re.I)
    if match:
        return sanitize_filename(unquote(match.group(1)))
    match = re.search(r'filename="?([^";]+)"?', disposition, re.I)
    if match:
        return sanitize_filename(unquote(match.group(1)))
    return f"{fallback}.{extension.lstrip('.')}"


def get_nested_value(data: dict[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_nested_value(data: dict[str, Any], key: str, value: Any) -> None:
    if key in data:
        data[key] = value
        return
    current = data
    parts = key.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def first_configured_key(cfg: dict[str, Any], name: str) -> str:
    keys = ((cfg.get("api") or {}).get("parameter_keys") or {})
    value = keys.get(name, "")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def guess_key(payload: dict[str, Any], candidates: list[str]) -> str:
    lowered = {key.lower(): key for key in payload.keys()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for key in payload.keys():
        low = key.lower()
        if any(candidate.lower() in low for candidate in candidates):
            return key
    return ""


def infer_parameter_keys(payload: dict[str, Any], cfg: dict[str, Any]) -> dict[str, str]:
    return {
        "device_ids": first_configured_key(cfg, "device_ids")
        or guess_key(payload, ["deviceIds", "deviceId", "meterIds", "meterId", "ids", "id", "nodeIds", "pointIds"]),
        "excel_names": first_configured_key(cfg, "excel_names")
        or guess_key(payload, ["excelNames", "equipmentNames", "deviceNames", "names", "name"]),
        "start_time": first_configured_key(cfg, "start_time")
        or guess_key(payload, ["startTime", "beginTime", "startDate", "beginDate", "sTime", "stime", "start"]),
        "end_time": first_configured_key(cfg, "end_time")
        or guess_key(payload, ["endTime", "stopTime", "finishTime", "endDate", "eTime", "etime", "end"]),
        "interval": first_configured_key(cfg, "interval")
        or guess_key(payload, ["interval", "dateType", "timeType", "queryType", "type", "step", "granularity"]),
        # exportEquipmentHistory has no separate "without chart" parameter in the captured
        # request. Do not guess here, otherwise "type" (hour/day/month/year) gets overwritten.
        "export_type": first_configured_key(cfg, "export_type"),
    }


def node_value(node: dict[str, Any], cfg: dict[str, Any]) -> str:
    api_cfg = cfg.get("api") or {}
    keys = api_cfg.get("device_value_keys") or []
    attrs = node.get("attrs") or {}
    raw = node.get("raw") or {}
    for key in keys:
        if key in node and node[key]:
            return str(node[key])
        if key in raw and raw[key]:
            return str(raw[key])
        if key in attrs and attrs[key]:
            return str(attrs[key])
    candidates = node.get("valueCandidates") or []
    for candidate in candidates:
        if candidate and re.fullmatch(r"\d+", str(candidate)):
            return str(candidate)
    for candidate in candidates:
        text = str(candidate or "")
        match = re.search(r"(?:^|[^\d])(\d{2,})(?:[^\d]|$)", text)
        if match:
            return match.group(1)
    return str(node.get("pathText") or node.get("label") or "")


def node_values(nodes: list[dict[str, Any]], cfg: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for node in nodes:
        value = node_value(node, cfg)
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def node_names(nodes: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for node in nodes:
        name = str(node.get("label") or node.get("pathText") or "").strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def update_payload_for_export(
    payload: dict[str, Any],
    keys: dict[str, str],
    cfg: dict[str, Any],
    batch: ExportBatch,
    start_time: str,
    end_time: str,
    interval_key: str,
) -> dict[str, Any]:
    updated = dict(payload)
    device_key = keys.get("device_ids")
    if not device_key:
        raise RuntimeError("无法识别设备参数名。请运行 capture_api.bat 后在 config.yaml 的 api.parameter_keys.device_ids 填入真实参数名。")

    values = node_values(batch.nodes, cfg)
    if (cfg.get("api") or {}).get("require_numeric_device_ids", False):
        bad_values = [value for value in values if not re.fullmatch(r"\d+", str(value))]
        if bad_values:
            raise RuntimeError(
                "设备 ID 提取失败，拿到的不是数字 ID: "
                + ", ".join(bad_values[:10])
                + "。请把这段报错和所选目录发给我。"
            )
    current = get_nested_value(updated, device_key)
    if isinstance(current, list):
        device_value: Any = values
    else:
        device_value = ",".join(values)
    set_nested_value(updated, device_key, device_value)

    excel_names_key = keys.get("excel_names")
    if excel_names_key:
        names = node_names(batch.nodes)
        current_names = get_nested_value(updated, excel_names_key)
        if isinstance(current_names, list):
            set_nested_value(updated, excel_names_key, names)
        else:
            set_nested_value(updated, excel_names_key, ",".join(names))

    if keys.get("start_time"):
        set_nested_value(updated, keys["start_time"], start_time)
    if keys.get("end_time"):
        set_nested_value(updated, keys["end_time"], end_time)
    if keys.get("interval"):
        interval_values = (cfg.get("api") or {}).get("interval_values") or {}
        set_nested_value(updated, keys["interval"], interval_values.get(interval_key, interval_key))
    if keys.get("export_type"):
        set_nested_value(updated, keys["export_type"], (cfg.get("api") or {}).get("export_without_chart_value", "不含图表"))
    return updated


def requests_session_from_context(context: Any) -> requests.Session:
    session = requests.Session()
    for cookie in context.cookies():
        session.cookies.set(
            cookie["name"],
            cookie["value"],
            domain=cookie.get("domain"),
            path=cookie.get("path", "/"),
        )
    return session


def clean_headers(headers: dict[str, str]) -> dict[str, str]:
    blocked = {"host", "content-length", "cookie", "origin"}
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def send_captured_request(
    session: requests.Session,
    capture: dict[str, Any],
    payload: dict[str, Any],
) -> requests.Response:
    method = str(capture.get("method", "GET")).upper()
    url = capture["url"]
    headers = clean_headers(capture.get("headers") or {})
    body_format = capture.get("body_format") or guess_body_format(headers, capture.get("post_data"))

    kwargs: dict[str, Any] = {"headers": headers, "timeout": 180}
    if method == "GET":
        parts = urlsplit(url)
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
        kwargs["params"] = payload
    elif body_format == "json":
        kwargs["json"] = payload
    else:
        kwargs["data"] = payload

    response = session.request(method, url, **kwargs)
    response.raise_for_status()
    return response


def request_export_file(
    session: requests.Session,
    capture: dict[str, Any],
    payload: dict[str, Any],
    batch: ExportBatch,
    batches_dir: Path,
    batch_index: int,
    extension: str,
) -> Path:
    response = send_captured_request(session, capture, payload)
    fallback = sanitize_filename(f"batch_{batch_index:03d}_{batch.label}", 140)
    filename = response_filename(dict(response.headers), fallback, extension)
    suffix = Path(filename).suffix or f".{extension.lstrip('.')}"
    target = batches_dir / f"batch_{batch_index:03d}{suffix}"
    target.write_bytes(response.content)
    return target


def save_debug_payload(
    cfg: dict[str, Any],
    batch_index: int,
    batch: ExportBatch,
    prep_payload: dict[str, Any] | None,
    export_payload: dict[str, Any],
) -> None:
    if not ((cfg.get("api") or {}).get("debug_payloads", False)):
        return
    debug_dir = PROJECT_DIR / "debug_payloads"
    debug_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "batch_index": batch_index,
        "label": batch.label,
        "leaf_count": batch.leaf_count,
        "node_values": node_values(batch.nodes, cfg),
        "node_names": node_names(batch.nodes),
        "node_details": [
            {
                "label": node.get("label"),
                "pathText": node.get("pathText"),
                "valueCandidates": node.get("valueCandidates"),
                "raw": node.get("raw"),
                "attrs": node.get("attrs"),
            }
            for node in batch.nodes
        ],
        "prep_payload": prep_payload,
        "export_payload": export_payload,
    }
    with (debug_dir / f"batch_{batch_index:03d}.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def run_api_exports(
    context: Any,
    cfg: dict[str, Any],
    batches: list[ExportBatch],
    start_time: str,
    end_time: str,
    interval_key: str,
    batches_dir: Path,
) -> list[Path]:
    capture = load_capture_file(cfg)
    use_prep = bool((cfg.get("api") or {}).get("use_prep_request", False))
    prep_capture = capture.get("prep_request") if use_prep and isinstance(capture.get("prep_request"), dict) else None
    parameter_capture = prep_capture or capture
    base_payload = parse_request_payload(parameter_capture)
    keys = infer_parameter_keys(base_payload, cfg)
    missing = [name for name in ("device_ids", "start_time", "end_time") if not keys.get(name)]
    if missing:
        raise RuntimeError(
            "接口参数名自动识别不完整: "
            + ", ".join(missing)
            + "。请在 config.yaml 的 api.parameter_keys 中补充。"
        )

    print("接口参数映射:")
    for name, key in keys.items():
        print(f"  {name}: {key or '(未设置)'}")

    session = requests_session_from_context(context)
    downloaded: list[Path] = []
    extension = str(cfg.get("output_extension", "xlsx")).lstrip(".")
    export_payload = parse_request_payload(capture)
    export_keys = infer_parameter_keys(export_payload, cfg)
    for index, batch in enumerate(batches, start=1):
        print(f"\n接口导出第 {index}/{len(batches)} 批: {batch.label}，约 {batch.leaf_count} 台")
        payload = update_payload_for_export(base_payload, keys, cfg, batch, start_time, end_time, interval_key)
        prep_payload_for_debug = None
        if prep_capture:
            send_captured_request(session, prep_capture, payload)
            prep_payload_for_debug = payload
            payload_for_download = export_payload
            if export_payload and export_keys.get("device_ids"):
                payload_for_download = update_payload_for_export(
                    export_payload,
                    export_keys,
                    cfg,
                    batch,
                    start_time,
                    end_time,
                    interval_key,
                )
        else:
            payload_for_download = payload
        save_debug_payload(cfg, index, batch, prep_payload_for_debug, payload_for_download)
        file = request_export_file(session, capture, payload_for_download, batch, batches_dir, index, extension)
        downloaded.append(file)
        print(f"已下载: {file.name}")
    return downloaded


def inspect_page(page: Page) -> None:
    open_matching_iframe_as_page(page, load_config(CONFIG_PATH))
    scopes: list[Scope] = [page.main_frame]
    scopes.extend(frame for frame in page.frames if frame is not page.main_frame)
    for idx, scope in enumerate(scopes):
        try:
            info = scope.evaluate(INSPECT_JS)
        except PlaywrightError as exc:
            print(f"\n[{idx}] {frame_label(scope)} 无法读取: {exc}")
            continue

        print(f"\n[{idx}] {frame_label(scope)}")
        print(f"标题: {info.get('title')!r}")
        print(f"地址: {info.get('url')!r}")

        print("iframe：")
        for item in info["iframes"]:
            print(f"  #{item['index']}: src={item['src']!r}, name={item['name']!r}, id={item['id']!r}, class={item['class']!r}")

        print("输入框：")
        for item in info["inputs"]:
            print(f"  #{item['index']}: placeholder={item['placeholder']!r}, value={item['value']!r}, class={item['class']!r}")

        print("按钮/可点击文本：")
        for text in info["buttons"]:
            print(f"  - {text}")

        print("可能的设备树：")
        for tree in info["trees"]:
            print(f"  #{tree['index']}: class={tree['class']!r}, text={tree['text']!r}")


def request_record(request: Any) -> dict[str, Any]:
    return {
        "url": request.url,
        "method": request.method,
        "headers": dict(request.headers),
        "post_data": request.post_data,
        "resource_type": request.resource_type,
    }


def payload_signal_score(record: dict[str, Any]) -> int:
    payload = parse_request_payload(record)
    if not payload:
        return 0
    keys = " ".join(payload.keys()).lower()
    values = " ".join(str(value) for value in payload.values()).lower()
    text = f"{keys} {values}"
    score = len(payload)
    for word in ["device", "meter", "ids", "id", "start", "begin", "end", "time", "date", "type"]:
        if word in text:
            score += 10
    return score


STATIC_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".css",
    ".js",
    ".map",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}


def is_static_asset(record: dict[str, Any]) -> bool:
    path = urlsplit(str(record.get("url", ""))).path.lower()
    if any(path.endswith(ext) for ext in STATIC_EXTENSIONS):
        return True
    resource_type = str(record.get("resource_type", "")).lower()
    return resource_type in {"image", "stylesheet", "script", "font"}


def score_capture_candidate(record: dict[str, Any]) -> int:
    if is_static_asset(record) and not record.get("matched_download"):
        return -1000
    url = str(record.get("url", "")).lower()
    method = str(record.get("method", "")).upper()
    post_data = str(record.get("post_data") or "")
    score = 0
    if record.get("matched_download"):
        score += 100
    if method == "POST":
        score += 20
    if any(word in url for word in ["export", "excel", "download", "daochu", "file", "xls", "xlsx", "导出"]):
        score += 40
    if any(word in post_data.lower() for word in ["export", "excel", "chart", "device", "meter", "start", "begin", "end", "ids"]):
        score += 15
    if post_data:
        score += 10
    return score


def write_capture_candidates(records: list[dict[str, Any]], path: Path) -> None:
    simplified = []
    for record in records:
        item = {
            "score": score_capture_candidate(record),
            "url": record.get("url"),
            "method": record.get("method"),
            "resource_type": record.get("resource_type"),
            "post_data": record.get("post_data"),
            "matched_download": record.get("matched_download", False),
        }
        simplified.append(item)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(simplified, f, allow_unicode=True, sort_keys=False)


def choose_parameter_source(records: list[dict[str, Any]], export_record: dict[str, Any]) -> dict[str, Any] | None:
    export_seq = int(export_record.get("seq", 10**9))
    candidates = []
    for record in records:
        if record is export_record:
            continue
        if is_static_asset(record):
            continue
        if int(record.get("seq", 0)) > export_seq:
            continue
        score = payload_signal_score(record)
        if score <= 0:
            continue
        candidates.append((score, record))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def run_capture_api(args: argparse.Namespace) -> None:
    cfg = load_config(CONFIG_PATH)
    profile_dir, downloads_dir, _, _ = ensure_dirs(cfg)
    capture_path = PROJECT_DIR / cfg.get("api_capture_file", "api_capture.yaml")
    candidates_path = PROJECT_DIR / "api_capture_candidates.yaml"

    captured_requests: list[dict[str, Any]] = []
    attachment_urls: set[str] = set()
    download_urls: set[str] = set()

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            channel=cfg.get("browser_channel") or None,
            headless=False,
            slow_mo=int(cfg.get("slow_mo_ms", 0)),
            accept_downloads=True,
            downloads_path=str(downloads_dir),
        )
        def on_request(request: Any) -> None:
            try:
                record = request_record(request)
                record["seq"] = len(captured_requests)
                captured_requests.append(record)
            except PlaywrightError:
                pass

        def on_response(response: Any) -> None:
            try:
                headers = response.headers
                content_type = headers.get("content-type", "").lower()
                disposition = headers.get("content-disposition", "").lower()
                if "attachment" in disposition or "excel" in content_type or "spreadsheet" in content_type:
                    attachment_urls.add(response.url)
            except PlaywrightError:
                pass

        def on_download(download: Any) -> None:
            try:
                download_urls.add(download.url)
            except PlaywrightError:
                pass

        def attach_page_events(event_page: Page) -> None:
            event_page.on("download", on_download)

        context.on("request", on_request)
        context.on("response", on_response)
        context.on("page", attach_page_events)

        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(15000)
        for existing_page in context.pages:
            attach_page_events(existing_page)

        try:
            open_or_login(page, cfg)
            open_matching_iframe_as_page(page, cfg)
            print("\n请在浏览器中手动完成一次最小范围导出，并在弹窗中选择“不含图表”。")
            print("下载开始或完成后，回到这里按回车，脚本会保存接口请求样本。")
            input("手动导出完成后按回车继续...")
            page.wait_for_timeout(2000)

            candidates = []
            for record in captured_requests:
                if record["url"] in attachment_urls or record["url"] in download_urls:
                    record["matched_download"] = True
                    candidates.append(record)
            if not candidates:
                candidates = [
                    record
                    for record in captured_requests
                    if not is_static_asset(record) and score_capture_candidate(record) > 0
                ]
                candidates = sorted(candidates, key=score_capture_candidate, reverse=True)[:10]
            else:
                candidates = sorted(candidates, key=score_capture_candidate, reverse=True)

            if not candidates:
                write_capture_candidates(
                    sorted(captured_requests, key=score_capture_candidate, reverse=True)[:30],
                    candidates_path,
                )
                raise RuntimeError(
                    f"没有捕获到可用的导出接口请求，已保存候选请求: {candidates_path}。"
                    "请确认手动导出确实触发了下载。"
                )

            best = candidates[0]
            if is_static_asset(best):
                write_capture_candidates(candidates, candidates_path)
                raise RuntimeError(
                    f"捕获到的最佳候选仍是静态资源: {best['url']}。"
                    f"候选请求已保存: {candidates_path}"
                )
            best["body_format"] = guess_body_format(best.get("headers", {}), best.get("post_data"))
            best["captured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            best["payload_keys"] = list(parse_request_payload(best).keys())
            prep = choose_parameter_source(captured_requests, best)
            if prep:
                prep["body_format"] = guess_body_format(prep.get("headers", {}), prep.get("post_data"))
                prep["payload_keys"] = list(parse_request_payload(prep).keys())
                best["prep_request"] = prep
            write_capture_file(best, capture_path)

            print(f"\n已保存接口样本: {capture_path}")
            print(f"方法: {best['method']}")
            print(f"地址: {best['url']}")
            print(f"请求体格式: {best['body_format']}")
            print("下载接口捕获到的参数名:")
            for key in best["payload_keys"]:
                print(f"  - {key}")
            if prep:
                print("\n查询/参数接口:")
                print(f"方法: {prep['method']}")
                print(f"地址: {prep['url']}")
                print(f"请求体格式: {prep['body_format']}")
                print("查询/参数接口捕获到的参数名:")
                for key in prep["payload_keys"]:
                    print(f"  - {key}")
            print("\n下一步运行 run.bat。若提示参数名识别失败，把上面的参数名发给我或填入 config.yaml 的 api.parameter_keys。")
        finally:
            context.close()


def run_export(args: argparse.Namespace) -> None:
    cfg = load_config(CONFIG_PATH)
    profile_dir, downloads_dir, batches_dir, output_dir = ensure_dirs(cfg)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            channel=cfg.get("browser_channel") or None,
            headless=bool(cfg.get("headless", False)),
            slow_mo=int(cfg.get("slow_mo_ms", 0)),
            accept_downloads=True,
            downloads_path=str(downloads_dir),
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(15000)

        try:
            open_or_login(page, cfg)
            open_matching_iframe_as_page(page, cfg)
            if args.inspect:
                inspect_page(page)
                return

            scope, tree_kind = find_app_scope(page, cfg)
            round_index = 1
            while True:
                print(f"\n========== 第 {round_index} 次导出 ==========")
                device_name, selected_choices = choose_tree_choices(scope, page, cfg, tree_kind)
                if not selected_choices:
                    raise RuntimeError("没有选中任何设备。")

                start_time = ask("请输入开始时间", "2026-04-28 00:00:00")
                end_time = ask("请输入截止时间", "2026-04-29 00:00:00")
                interval_raw = ask("请输入查询间隔（时/日/月/年）", "时")
                interval_key, interval_label = normalize_interval(interval_raw, cfg)

                max_devices = int(cfg.get("max_devices_per_query", 100))
                batches = make_export_batches(selected_choices, max_devices)
                total_leaves = sum(len(choice.leaves) for choice in selected_choices)
                directory_batches = sum(1 for batch in batches if not batch.uses_leaf_devices)
                leaf_batches = sum(1 for batch in batches if batch.uses_leaf_devices)
                print(f"\n共选择约 {total_leaves} 台设备，将分 {len(batches)} 批导出。")
                print(f"目录批次: {directory_batches}，单设备拆分批次: {leaf_batches}。")

                if str(cfg.get("export_mode", "browser")).lower() == "api":
                    downloaded = run_api_exports(context, cfg, batches, start_time, end_time, interval_key, batches_dir)
                else:
                    downloaded = []
                    for index, batch in enumerate(batches, start=1):
                        print(f"\n正在处理第 {index}/{len(batches)} 批: {batch.label}，约 {batch.leaf_count} 台")
                        set_time_inputs(scope, page, cfg, start_time, end_time)
                        set_interval(scope, page, interval_label)
                        select_devices(scope, page, cfg, tree_kind, batch.nodes)
                        query(scope, page, cfg)
                        file = export_download(scope, page, cfg, batches_dir, index)
                        downloaded.append(file)
                        print(f"已下载: {file.name}")

                ext = str(cfg.get("output_extension", "xlsx")).lstrip(".")
                base = f"{sanitize_filename(device_name)}_{sanitize_time_for_filename(start_time)}_{sanitize_time_for_filename(end_time)}"
                output_path = output_dir / f"{base}.{ext}"
                merge_files(downloaded, output_path)

                if not bool(cfg.get("keep_batch_files", True)):
                    shutil.rmtree(batches_dir, ignore_errors=True)
                    batches_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n完成，合并文件已生成: {output_path}")
                answer = ask("是否继续导出下一次？输入 y 继续，直接回车或输入 n 结束", "n").strip().lower()
                if answer not in {"y", "yes", "是", "继续"}:
                    print("已结束导出。")
                    break
                round_index += 1
        finally:
            context.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="冷热量表历史记录分批导出工具")
    parser.add_argument("--inspect", action="store_true", help="登录并进入页面后打印控件信息，不执行导出")
    parser.add_argument("--capture-api", action="store_true", help="手动导出一次，抓取后端导出接口样本")
    args = parser.parse_args()

    try:
        if args.capture_api:
            run_capture_api(args)
        else:
            run_export(args)
        return 0
    except KeyboardInterrupt:
        print("\n已取消。")
        return 130
    except Exception as exc:
        print(f"\n运行失败: {exc}", file=sys.stderr)
        print("可先运行 python export_tool.py --inspect 查看页面控件，再调整 config.yaml。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
