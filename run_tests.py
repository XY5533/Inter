"""
测试运行器 - CLI入口
支持单次测试、回归测试、查看报告
"""

import os
import sys
import json
import argparse

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.evaluator import Evaluator, run_test_suite
from tests.report_generator import generate_single_report, generate_regression_report, load_results


def cmd_run(args):
    """执行单次测试"""
    results = run_test_suite(rebuild_kb=args.rebuild)
    report = generate_single_report(
        [r.to_dict() for r in results],
        output_dir=args.output_dir or "test_results",
    )
    print("\n" + report[:500] + "\n...")


def cmd_regression(args):
    """执行回归测试"""

    evaluator = Evaluator()
    evaluator.load_test_cases()

    # 第1次：检查是否有基线数据
    results_dir = args.output_dir or "test_results"
    os.makedirs(results_dir, exist_ok=True)

    # 查找上一次的测试结果
    existing_results = sorted(
        [f for f in os.listdir(results_dir) if f.startswith("results_") and f.endswith(".json")]
    )

    if existing_results and not args.force_baseline:
        # 使用最近一次结果作为基线
        baseline_path = os.path.join(results_dir, existing_results[-1])
        baseline_data = load_results(baseline_path)

        # 构建基线摘要
        passed = sum(1 for r in baseline_data if r["passed"])
        baseline = {
            "label": args.label_a or f"基线 ({existing_results[-1]})",
            "timestamp": baseline_data[0].get("timestamp", "未知") if baseline_data else "未知",
            "total": len(baseline_data),
            "passed": passed,
            "failed_count": len(baseline_data) - passed,
            "failed_cases": [r for r in baseline_data if not r["passed"]],
            "pass_rate": round(passed / len(baseline_data) * 100, 1) if baseline_data else 0,
            "avg_latency_ms": round(sum(r["latency_ms"] for r in baseline_data) / len(baseline_data), 1) if baseline_data else 0,
            "total_time_ms": round(sum(r["latency_ms"] for r in baseline_data), 1) if baseline_data else 0,
            "results": baseline_data,
        }

        print(f"  基线: {baseline['label']} (通过率 {baseline['pass_rate']}%)")
    else:
        # 无基线，先跑一次作为基线
        print("  未找到基线数据，先运行一次作为基线...")
        results = evaluator.evaluate_all(rebuild_kb=args.rebuild)
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(results_dir, f"results_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
        print(f"  基线已保存: {json_path}")
        print("  请修改Prompt或模型后，再次运行回归测试进行对比。")
        return

    # 第2次：运行当前版本
    print("\n  运行当前版本...")
    current_results = evaluator.evaluate_all(rebuild_kb=args.rebuild)

    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"results_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in current_results], f, ensure_ascii=False, indent=2)
    print(f"  当前结果已保存: {json_path}")

    current_passed = sum(1 for r in current_results if r.passed)
    current = {
        "label": args.label_b or f"当前 ({ts})",
        "timestamp": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(current_results),
        "passed": current_passed,
        "failed_count": len(current_results) - current_passed,
        "failed_cases": [r for r in current_results if not r.passed],
        "pass_rate": round(current_passed / len(current_results) * 100, 1) if current_results else 0,
        "avg_latency_ms": round(sum(r.latency_ms for r in current_results) / len(current_results), 1) if current_results else 0,
        "total_time_ms": round(sum(r.latency_ms for r in current_results), 1) if current_results else 0,
        "results": [r.to_dict() for r in current_results],
    }

    # 生成对比报告
    report = generate_regression_report(baseline, current, output_dir=results_dir)
    print("\n" + report[:800] + "\n...")


def cmd_list(args):
    """列出测试用例"""
    evaluator = Evaluator()
    count = evaluator.load_test_cases(args.file)

    print(f"\n📋 测试用例集 ({count} 条)\n")
    print(f"{'ID':<10} {'分类':<20} {'问题':<35} {'期望关键词':<25}")
    print("-" * 90)

    for case in evaluator.test_cases:
        q = case.question[:30]
        kw = ", ".join(case.expected_keywords[:3])
        print(f"{case.id:<10} {case.category:<20} {q:<35} {kw:<25}")


def cmd_report(args):
    """从已有结果生成报告"""
    results = load_results(args.input)
    report = generate_single_report(results, output_dir=args.output_dir or "test_results")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="AI应用评估测试工具")
    sub = parser.add_subparsers(dest="command", help="子命令")

    # run
    p_run = sub.add_parser("run", help="执行单次测试")
    p_run.add_argument("--rebuild", action="store_true", help="强制重建知识库")
    p_run.add_argument("--output-dir", default="test_results", help="输出目录")

    # regression
    p_reg = sub.add_parser("regression", help="回归测试（对比前后差异）")
    p_reg.add_argument("--rebuild", action="store_true", help="强制重建知识库")
    p_reg.add_argument("--label-a", default=None, help="基线版本名称")
    p_reg.add_argument("--label-b", default=None, help="当前版本名称")
    p_reg.add_argument("--output-dir", default="test_results", help="输出目录")
    p_reg.add_argument("--force-baseline", action="store_true", help="忽略已有结果，新建基线")

    # list
    p_list = sub.add_parser("list", help="列出测试用例")
    p_list.add_argument("--file", default=None, help="测试用例文件路径")

    # report
    p_rpt = sub.add_parser("report", help="从已有结果生成报告")
    p_rpt.add_argument("input", help="结果JSON文件路径")
    p_rpt.add_argument("--output-dir", default="test_results", help="输出目录")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "regression":
        cmd_regression(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
