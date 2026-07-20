"""
报告生成器
生成 Markdown 格式的测试报告，支持单次报告和回归对比报告
"""

import os
import json
import datetime
from typing import List, Dict, Any


def _status_icon(r: dict) -> str:
    return "✅" if r["passed"] else "❌"


def _bar(value: float, total: float = 100.0, width: int = 20) -> str:
    """生成简单进度条"""
    filled = int((value / total) * width) if total > 0 else 0
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


def generate_single_report(
    results_data: List[dict],
    output_dir: str = None,
) -> str:
    """
    生成单次测试 Markdown 报告
    results_data: evaluator 输出的结果字典列表
    """
    total = len(results_data)
    passed = sum(1 for r in results_data if r["passed"])
    failed = [r for r in results_data if not r["passed"]]
    pass_rate = round(passed / total * 100, 1) if total else 0
    avg_latency = round(sum(r["latency_ms"] for r in results_data) / total, 1) if total else 0
    total_time = round(sum(r["latency_ms"] for r in results_data) / 1000, 1)
    avg_token = round(sum(r["token_estimate"] for r in results_data) / total) if total else 0

    # 按分类统计
    categories = {}
    for r in results_data:
        cat = r.get("category", "未分类")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# 📊 AI应用评估测试报告")
    lines.append("")
    lines.append(f"**生成时间**: {ts}  ")
    lines.append(f"**测试用例数**: {total}  ")
    lines.append(f"**通过/失败**: {passed} / {len(failed)}  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 汇总卡片
    lines.append("## 📈 总体指标")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 用例总数 | {total} |")
    lines.append(f"| 通过数 | {passed} |")
    lines.append(f"| 失败数 | {len(failed)} |")
    lines.append(f"| 通过率 | {pass_rate}% |")
    lines.append(f"| 平均响应时间 | {avg_latency} ms |")
    lines.append(f"| 总耗时 | {total_time} s |")
    lines.append(f"| 平均估算Token | {avg_token} |")
    lines.append("")
    lines.append(f"通过率: {_bar(pass_rate, 100)} {pass_rate}%")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 分类统计
    lines.append("## 📂 分类统计")
    lines.append("")
    lines.append("| 分类 | 总数 | 通过 | 通过率 |")
    lines.append("|------|------|------|--------|")
    for cat, stats in sorted(categories.items()):
        cat_rate = round(stats["passed"] / stats["total"] * 100, 1)
        lines.append(f"| {cat} | {stats['total']} | {stats['passed']} | {cat_rate}% |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 明细表格
    lines.append("## 📋 用例明细")
    lines.append("")
    lines.append("| 状态 | ID | 分类 | 问题 | 耗时(ms) | 关键词命中 | 异常检查 | 长度检查 | 说明 |")
    lines.append("|------|-----|------|------|----------|------------|----------|----------|------|")

    for r in sorted(results_data, key=lambda x: x["case_id"]):
        kw_hit = "✅" if r.get("keyword_hit", False) or not r.get("expected_keywords") else "❌"
        neg_pass = "✅" if r.get("negative_check_pass") else "❌"
        len_pass = "✅" if r.get("length_check_pass") else "❌"
        question_short = r["question"][:25] + "..." if len(r["question"]) > 25 else r["question"]

        lines.append(
            f"| {_status_icon(r)} | {r['case_id']} "
            f"| {r.get('category', '-')} "
            f"| {question_short} "
            f"| {r.get('latency_ms', 0)} "
            f"| {kw_hit} "
            f"| {neg_pass} "
            f"| {len_pass} "
            f"| {'检索到' + str(r.get('retrieved_docs', 0)) + '篇文档' if r.get('retrieved_docs',0) > 0 else '无匹配文档'}"
            f" |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # 失败用例详情
    if failed:
        lines.append("## ❌ 失败用例详情")
        lines.append("")
        for r in failed:
            lines.append(f"### {r['case_id']}: {r['question']}")
            lines.append("")
            lines.append(f"- **分类**: {r.get('category', '-')}")
            lines.append(f"- **耗时**: {r['latency_ms']} ms")
            lines.append(f"- **检索文档数**: {r.get('retrieved_docs', 0)}")

            if r.get("expected_keywords"):
                lines.append(f"- **期望关键词**: `{'`, `'.join(r['expected_keywords'])}`")
                lines.append(f"- **关键词命中率**: {r.get('keyword_hit_rate', 0) * 100:.0f}%")

            if r.get("errors"):
                lines.append(f"- **异常信息**:")
                for e in r["errors"]:
                    lines.append(f"  - {e}")

            lines.append("")
            lines.append("```")
            lines.append(f"实际回答:\n{r.get('actual_response', '')[:300]}")
            lines.append("```")
            lines.append("")

    # 保存
    report = "\n".join(lines)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"report_{ts_file}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  [OK] 报告已保存: {path}")

    return report


def generate_regression_report(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    output_dir: str = None,
) -> str:
    """
    生成回归测试对比报告
    baseline: 上一次测试结果
    current:  本次测试结果
    """
    lines = []
    lines.append("# 🔄 回归测试对比报告")
    lines.append("")

    lines.append(f"**基线版本**: {baseline['label']} ({baseline['timestamp']})")
    lines.append(f"**当前版本**: {current['label']} ({current['timestamp']})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 总体对比
    lines.append("## 📈 总体指标对比")
    lines.append("")
    lines.append("| 指标 | 基线 | 当前 | 变化 |")
    lines.append("|------|------|------|------|")

    def _delta(v1, v2, suffix=""):
        d = v2 - v1
        sign = "+" if d > 0 else ""
        return f"{sign}{d}{suffix}"

    lines.append(f"| 通过率 | {baseline['pass_rate']}% | {current['pass_rate']}% | {_delta(baseline['pass_rate'], current['pass_rate'], 'pp')} |")
    lines.append(f"| 通过数 | {baseline['passed']}/{baseline['total']} | {current['passed']}/{current['total']} | {_delta(baseline['passed'], current['passed'])} |")
    lines.append(f"| 平均响应时间 | {baseline['avg_latency_ms']} ms | {current['avg_latency_ms']} ms | {_delta(baseline['avg_latency_ms'], current['avg_latency_ms'])} ms |")
    lines.append(f"| 总耗时 | {baseline['total_time_ms']/1000:.1f} s | {current['total_time_ms']/1000:.1f} s | {_delta(round(baseline['total_time_ms']/1000,1), round(current['total_time_ms']/1000,1))} s |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 逐个用例对比
    lines.append("## 📋 用例级对比")
    lines.append("")

    # 建立 case_id → result 的映射
    base_map = {r["case_id"]: r for r in baseline["results"]}
    curr_map = {r["case_id"]: r for r in current["results"]}

    all_ids = sorted(set(list(base_map.keys()) + list(curr_map.keys())))

    lines.append("| ID | 问题 | 基线状态 | 当前状态 | 耗时变化 | 关键词命中变化 |")
    lines.append("|-----|------|----------|----------|----------|----------------|")

    for cid in all_ids:
        br = base_map.get(cid)
        cr = curr_map.get(cid)

        question = (cr or br)["question"][:22] + "..." if len((cr or br)["question"]) > 22 else (cr or br)["question"]
        b_status = _status_icon(br) + "通过" if br and br["passed"] else (_status_icon(br) + "失败" if br else "—")
        c_status = _status_icon(cr) + "通过" if cr and cr["passed"] else (_status_icon(cr) + "失败" if cr else "—")

        b_lat = br["latency_ms"] if br else 0
        c_lat = cr["latency_ms"] if cr else 0
        lat_delta = f"{'+' if c_lat - b_lat > 0 else ''}{c_lat - b_lat:.0f}ms" if br else "—"

        b_kw = "✅" if br and (br.get("keyword_hit") or not br.get("expected_keywords")) else ("❌" if br else "—")
        c_kw = "✅" if cr and (cr.get("keyword_hit") or not cr.get("expected_keywords")) else ("❌" if cr else "—")

        kw_change = ""
        if br and cr:
            bv = br.get("keyword_hit", False) or not br.get("expected_keywords")
            cv = cr.get("keyword_hit", False) or not cr.get("expected_keywords")
            if bv and not cv:
                kw_change = "⬇️ 退步"
            elif not bv and cv:
                kw_change = "⬆️ 提升"
            elif bv == cv:
                kw_change = "➡️ 持平"
            else:
                kw_change = "—"

        lines.append(f"| {cid} | {question} | {b_status} | {c_status} | {lat_delta} | {kw_change} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 新增/消失的失败用例
    base_failed_ids = {r["case_id"] for r in baseline.get("failed_cases", [])}
    curr_failed_ids = {r["case_id"] for r in current.get("failed_cases", [])}

    new_failures = curr_failed_ids - base_failed_ids
    fixed = base_failed_ids - curr_failed_ids

    if new_failures:
        lines.append("### ⚠️ 新增失败用例")
        for cid in sorted(new_failures):
            lines.append(f"- **{cid}**: {curr_map[cid]['question']}")
        lines.append("")

    if fixed:
        lines.append("### ✅ 已修复用例")
        for cid in sorted(fixed):
            lines.append(f"- **{cid}**: {base_map[cid]['question']}")
        lines.append("")

    report = "\n".join(lines)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"regression_report_{ts_file}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  [OK] 回归报告已保存: {path}")

    return report


# ---- 快捷入口 ----

def load_results(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
